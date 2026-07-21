/* Pure form logic for the LLM-connections screen — no React, no fetch, so it tests as plain
   functions (the repo has no DOM test renderer). Provider matrix mirrors the desktop AI Console:
   openai/anthropic keyed · custom = OpenAI-compatible endpoint, baseUrl REQUIRED, key optional.
   Ollama is intentionally NOT a picker option — a local/remote Ollama is reached through `custom`
   at its own OpenAI-compatible /v1/chat/completions (same call the AI Console made). The backend
   still accepts an `ollama` provider, and providerFields keeps handling it so a legacy stored
   connection still edits correctly — only the NEW-connection dropdown drops it. */

export const PROVIDERS = ['openai', 'anthropic', 'custom'];

// The picker set, but with the row's CURRENT provider prepended when it isn't a picker option
// (a legacy `ollama` row being edited). Without this the provider Select's value has no matching
// option → it renders blank / coerces to the first option, silently rewriting provider on save.
export function providerOptions(current) {
  return current && !PROVIDERS.includes(current) ? [current, ...PROVIDERS] : PROVIDERS;
}

export function providerFields(provider) {
  if (provider === 'ollama') return { baseUrl: 'optional', apiKey: 'hidden' };   // legacy rows only
  if (provider === 'custom') return { baseUrl: 'required', apiKey: 'optional' };
  return { baseUrl: 'optional', apiKey: 'required' };          // openai | anthropic
}

export function canSave(form, mode) {
  if (!String(form.name || '').trim()) return false;
  const f = providerFields(form.provider);
  if (f.baseUrl === 'required' && !String(form.base_url || '').trim()) return false;
  // edit-mode key is always optional: blank means "keep the stored key"
  if (mode === 'create' && f.apiKey === 'required' && !String(form.api_key || '').trim()) return false;
  return true;
}

export function toPayload(form) {
  const p = {
    name: String(form.name || '').trim(),
    provider: form.provider,
    model: String(form.model || '').trim(),
    base_url: String(form.base_url || '').trim() || null,
  };
  const key = String(form.api_key || '');
  if (key) p.api_key = key;              // omitted = backend keeps the stored key
  return p;
}
