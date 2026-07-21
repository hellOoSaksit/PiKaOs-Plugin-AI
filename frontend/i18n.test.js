/* Mirrors the postgres plugin's DbChoice lesson: every key this plugin's screens pass to `t()`
   must be one the plugin owns — a key that silently lives in a Core pack renders as a raw string
   on any install without that residue. Packs use Core's wrapped shape
   { languageCode, lexiconCode, translations } (src/lib/i18n.jsx's plugin-pack merge loop skips
   any file missing `.translations`), so lookups go through `.translations`, not the pack root. */
import { describe, it, expect } from 'vitest';
import en from './i18n/en-formal.json';
import th from './i18n/th-formal.json';
import screenSrc from './LlmConfig.jsx?raw';
import { PROVIDERS } from './LlmConfig.logic.js';

// static keys: every t('...') literal in the screen source (catches single-quoted calls;
// dynamic concatenations are asserted explicitly below)
const staticKeys = [...screenSrc.matchAll(/\bt\('([^']+)'\)/g)].map((m) => m[1]);

// dynamic keys the screen builds by concatenation
const ROLES = ['engine', 'search', 'summarize', 'answer'];   // backend ROLES tuple; a new backend role falls back to its raw key until the pack learns it
// probe result categories the backend can return (llm_probe.categorize_*) — the screen builds
// t('llmcfg.test.cat.' + category); every one must have a localized message.
const TEST_CATEGORIES = ['ok', 'auth', 'not_found', 'rate_limit', 'timeout', 'connection', 'http', 'blocked', 'error'];
const dynamicKeys = [
  ...PROVIDERS.map((p) => 'llmcfg.provider.' + p),
  ...ROLES.flatMap((r) => ['llmcfg.role.' + r, 'llmcfg.role.' + r + '.desc']),
  ...TEST_CATEGORIES.map((c) => 'llmcfg.test.cat.' + c),
  'nav.llm-config',                                          // the sidebar renders "nav." + route id
  'route.llm-config.title',                                  // the topbar renders "route." + route id + ".title"
];

describe('AI plugin i18n packs own every key the screen uses', () => {
  for (const key of [...new Set([...staticKeys, ...dynamicKeys])]) {
    if (key === 'common.close') continue;                    // deliberately Core-owned (shared Modal vocab)
    it(`en+th own ${key}`, () => {
      expect(en.translations[key], `en missing ${key}`).toBeTypeOf('string');
      expect(th.translations[key], `th missing ${key}`).toBeTypeOf('string');
    });
  }
  it('found the static keys at all (regex sanity)', () => {
    expect(staticKeys.length).toBeGreaterThan(10);
  });
});
