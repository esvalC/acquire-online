/**
 * Bad-word filter (the `obscenity` library: a maintained English word list that
 * also catches look-alike spellings such as "sh1t").
 *   censor(text)  → the text with matched words replaced by asterisks
 *   isClean(text) → false when the text contains a matched word
 * Used for table chat, friend chat, news replies, display names and usernames.
 */
const {
  RegExpMatcher, TextCensor, englishDataset, englishRecommendedTransformers, asteriskCensorStrategy,
} = require('obscenity');

const matcher = new RegExpMatcher({ ...englishDataset.build(), ...englishRecommendedTransformers });
const censorer = new TextCensor().setStrategy(asteriskCensorStrategy());

function censor(text) {
  const s = String(text == null ? '' : text);
  try {
    const matches = matcher.getAllMatches(s, true);
    return matches.length ? censorer.applyTo(s, matches) : s;
  } catch { return s; }
}
function isClean(text) {
  try { return !matcher.hasMatch(String(text == null ? '' : text)); } catch { return true; }
}

module.exports = { censor, isClean };
