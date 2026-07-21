/* Pure form logic for the LLM-connections screen — no React, no fetch, so it tests as plain
   functions (the repo has no DOM test renderer). Mirrors the desktop AI Console's provider
   matrix so the two surfaces cannot drift: ollama keyless · openai/anthropic keyed ·
   custom = OpenAI-compatible endpoint, baseUrl REQUIRED, key optional. */

export const PROVIDERS = ['ollama', 'openai', 'anthropic', 'custom'];

export function providerFields(provider) {
  if (provider === 'ollama') return { baseUrl: 'optional', apiKey: 'hidden' };
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
