import { describe, it, expect } from 'vitest';
import { PROVIDERS, providerFields, canSave, toPayload } from './LlmConfig.logic.js';

describe('PROVIDERS', () => {
  it('matches the backend set, custom last', () => {
    expect(PROVIDERS).toEqual(['ollama', 'openai', 'anthropic', 'custom']);
  });
});

describe('providerFields', () => {
  it('ollama is keyless', () => {
    expect(providerFields('ollama')).toEqual({ baseUrl: 'optional', apiKey: 'hidden' });
  });
  it('openai/anthropic need a key', () => {
    expect(providerFields('openai')).toEqual({ baseUrl: 'optional', apiKey: 'required' });
    expect(providerFields('anthropic')).toEqual({ baseUrl: 'optional', apiKey: 'required' });
  });
  it('custom requires baseUrl, key optional', () => {
    expect(providerFields('custom')).toEqual({ baseUrl: 'required', apiKey: 'optional' });
  });
});

describe('canSave', () => {
  const base = { name: 'X', provider: 'ollama', model: '', base_url: '', api_key: '' };
  it('needs a name', () => {
    expect(canSave({ ...base, name: '  ' }, 'create')).toBe(false);
    expect(canSave(base, 'create')).toBe(true);
  });
  it('custom needs base_url', () => {
    expect(canSave({ ...base, provider: 'custom' }, 'create')).toBe(false);
    expect(canSave({ ...base, provider: 'custom', base_url: 'http://localhost:1234/v1/chat/completions' }, 'create')).toBe(true);
  });
  it('create-mode openai needs a key; edit-mode blank key = keep stored', () => {
    expect(canSave({ ...base, provider: 'openai' }, 'create')).toBe(false);
    expect(canSave({ ...base, provider: 'openai', api_key: 'sk-x' }, 'create')).toBe(true);
    expect(canSave({ ...base, provider: 'openai' }, 'edit')).toBe(true);
  });
});

describe('toPayload', () => {
  it('omits a blank key and nulls a blank base_url', () => {
    const p = toPayload({ name: ' A ', provider: 'ollama', model: ' m ', base_url: '', api_key: '' });
    expect(p).toEqual({ name: 'A', provider: 'ollama', model: 'm', base_url: null });
    expect('api_key' in p).toBe(false);
  });
  it('keeps a supplied key and base_url', () => {
    const p = toPayload({ name: 'A', provider: 'custom', model: '', base_url: ' http://h/v1/chat/completions ', api_key: 'k' });
    expect(p.base_url).toBe('http://h/v1/chat/completions');
    expect(p.api_key).toBe('k');
  });
});
