/**
 * Acquire – Auth helpers (profiles branch)
 * JWT session management + Twilio Verify OTP.
 *
 * In dev mode (no TWILIO_* env vars set), OTP codes are printed to the
 * server console instead of sent via SMS — no Twilio account needed to test.
 */

const jwt    = require('jsonwebtoken');
const crypto = require('crypto');

const JWT_SECRET  = process.env.JWT_SECRET  || 'dev-jwt-secret-change-in-production';
const JWT_EXPIRES = '30d';

/* ── Phone encryption (AES-256-GCM) ─────────────────────────── */
// Stores a reversible encrypted copy of the phone number so we can
// look up the phone by username for OTP login. The hash (phone_hash)
// is still used for uniqueness — this is only for OTP delivery.
const PHONE_KEY = Buffer.from(
  process.env.PHONE_ENCRYPT_KEY || '0'.repeat(64), 'hex'
);

function encryptPhone(phone) {
  const iv     = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', PHONE_KEY, iv);
  const enc    = Buffer.concat([cipher.update(phone, 'utf8'), cipher.final()]);
  const tag    = cipher.getAuthTag();
  return Buffer.concat([iv, tag, enc]).toString('base64');
}

function decryptPhone(b64) {
  const buf     = Buffer.from(b64, 'base64');
  const iv      = buf.subarray(0, 12);
  const tag     = buf.subarray(12, 28);
  const enc     = buf.subarray(28);
  const decipher = crypto.createDecipheriv('aes-256-gcm', PHONE_KEY, iv);
  decipher.setAuthTag(tag);
  return decipher.update(enc).toString('utf8') + decipher.final('utf8');
}

/* ── JWT ─────────────────────────────────────────────────────── */
function signToken(userId) {
  return jwt.sign({ sub: userId }, JWT_SECRET, { expiresIn: JWT_EXPIRES });
}

function verifyToken(token) {
  try { return jwt.verify(token, JWT_SECRET); }
  catch { return null; }
}

// Express middleware — attaches req.userId if a valid token is present
function requireAuth(req, res, next) {
  const header = req.headers.authorization || '';
  const token  = header.startsWith('Bearer ') ? header.slice(7) : null;
  if (!token) return res.status(401).json({ error: 'Not authenticated' });
  const payload = verifyToken(token);
  if (!payload) return res.status(401).json({ error: 'Invalid or expired session' });
  req.userId = payload.sub;
  next();
}

// Express middleware — requireAuth + checks is_admin flag in DB
function requireAdmin(req, res, next) {
  requireAuth(req, res, () => {
    const db   = require('./db');
    const user = db.findById(req.userId);
    if (!user || !user.is_admin) return res.status(403).json({ error: 'Admin access required' });
    next();
  });
}

/* ── In-memory OTP store ─────────────────────────────────────── */
// Maps phone → { code, expires, attempts }
// Never persisted to disk — intentional.
const otpStore = new Map();

const OTP_TTL_MS    = 10 * 60 * 1000; // 10 minutes
const MAX_ATTEMPTS  = 5;

function generateOtp() {
  return String(Math.floor(100000 + crypto.randomInt(900000)));
}

/* ── SMS via Twilio Messages (raw SMS, NOT Twilio Verify) ────── */
// We generate OTPs ourselves and store them in memory.
// Twilio only sees the phone number long enough to deliver the SMS —
// it does not log or track verifications on its end.
// Raw phone numbers are never written to our database (only encrypted).
let twilioClient = null;
const DEV_MODE   = !process.env.TWILIO_ACCOUNT_SID;

if (!DEV_MODE) {
  const twilio = require('twilio');
  twilioClient = twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN);
}

/**
 * Send an OTP to the given E.164 phone number.
 * Returns { ok: true } or { error: string }.
 */
async function sendOtp(e164) {
  const code = generateOtp();
  otpStore.set(e164, { code, expires: Date.now() + OTP_TTL_MS, attempts: 0 });

  if (DEV_MODE) {
    console.log(`\n[DEV OTP] Phone: ${e164}  Code: ${code}\n`);
    return { ok: true, dev: true };
  }

  try {
    await twilioClient.messages.create({
      body: `Your Acquire verification code: ${code}. Valid for 10 minutes.`,
      from: process.env.TWILIO_PHONE_NUMBER,
      to: e164,
    });
    return { ok: true };
  } catch (err) {
    otpStore.delete(e164); // don't leave a dangling code if send failed
    console.error('Twilio sendOtp error:', err.message);
    return { error: 'Failed to send SMS. Check the phone number and try again.' };
  }
}

/**
 * Verify an OTP for the given phone number.
 * Returns { valid: true } or { valid: false, error: string }.
 * Verification is always handled in-memory — Twilio is not involved here.
 */
async function verifyOtp(e164, code) {
  const entry = otpStore.get(e164);
  if (!entry)                        return { valid: false, error: 'No code sent to this number' };
  if (Date.now() > entry.expires)  { otpStore.delete(e164); return { valid: false, error: 'Code expired' }; }
  entry.attempts++;
  if (entry.attempts > MAX_ATTEMPTS) { otpStore.delete(e164); return { valid: false, error: 'Too many attempts' }; }
  if (entry.code !== String(code))   return { valid: false, error: 'Incorrect code' };
  otpStore.delete(e164);
  return { valid: true };
}

/* ── Phone normalisation ─────────────────────────────────────── */
// Very light normalisation — strips spaces/dashes, ensures + prefix.
// Full validation is handled by Twilio on their end.
function normalisePhone(raw) {
  const stripped = raw.replace(/[\s\-().]/g, '');
  if (!stripped.startsWith('+')) return '+1' + stripped; // default to US
  return stripped;
}

/**
 * Return a masked version of an E.164 number for display.
 * e.g. +14155551234 → +1 •••-•••-1234
 */
function maskPhone(e164) {
  const digits = e164.replace(/\D/g, '');
  const last4  = digits.slice(-4);
  const cc     = e164.startsWith('+1') ? '+1' : '+' + digits.slice(0, digits.length - 10);
  return `${cc} •••-•••-${last4}`;
}

module.exports = { signToken, verifyToken, requireAuth, requireAdmin, sendOtp, verifyOtp, normalisePhone, encryptPhone, decryptPhone, maskPhone };
