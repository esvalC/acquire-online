/**
 * ai/botWorker.js — runs searchBot off the main thread.
 *
 * The search is CPU-heavy (hundreds to thousands of simulated games per
 * move). Running it on the main Node thread would freeze every other game
 * and HTTP request while a bot thinks, so the server hands the work to this
 * worker and applies the returned action when it arrives.
 */
const { parentPort } = require('worker_threads');
const { decideSearchAction, lastSearchInfo } = require('./searchBot');

parentPort.on('message', ({ id, game, me, opts }) => {
  let action = null, error = null, info = null;
  try { action = decideSearchAction(game, me, opts); info = lastSearchInfo(); }
  catch (e) { error = e.stack || e.message; }
  parentPort.postMessage({ id, action, error, info });
});
