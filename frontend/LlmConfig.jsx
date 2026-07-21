/* AI plugin — admin LLM-connections screen (connections CRUD + activate + per-role binding).
   Thin client over the plugin's own /api/ai/llm/* routes; authz is server-side (llm.view /
   llm.manage / llm.assign) — the nav gate on llm.view is UI honesty, not enforcement.
   The API key is write-only: the UI only ever sees api_key_set. */
import React from 'react';
const { useState, useEffect, useCallback } = React;
import { Button, Field, Modal, Panel, PageHead, Table, Empty, Tooltip } from '../../components/ui';
import { Select } from '../../components/ui/Dropdown.jsx';
import { PROVIDERS, providerOptions, providerFields, canSave, toPayload } from './LlmConfig.logic.js';
import './llm-config.css';

const EMPTY_FORM = { name: '', provider: 'openai', model: '', base_url: '', api_key: '' };

export function LlmConfig({ ctx }) {
  const { t, api } = ctx;
  const [conns, setConns] = useState(null);          // null = loading
  const [roles, setRoles] = useState([]);
  const [error, setError] = useState(null);
  const [form, setForm] = useState(null);            // { mode: 'create'|'edit', id?, data }
  const [busy, setBusy] = useState(false);
  const [tests, setTests] = useState({});            // { [connId]: { loading?, ok, category, status } }

  const load = useCallback(() => {
    setError(null);
    Promise.all([api.raw('/ai/llm/connections'), api.raw('/ai/llm/roles')])
      .then(([cs, rs]) => {
        setConns(Array.isArray(cs) ? cs : []);
        setRoles(Array.isArray(rs) ? rs : []);
      })
      .catch((e) => { setConns([]); setError(e.message || 'error'); });
  }, [api]);
  useEffect(() => { load(); }, [load]);

  // every mutation: run → reload the lists → surface any failure in the banner
  const act = (fn) => {
    setBusy(true);
    fn().then(() => { setForm(null); load(); })
      .catch((e) => setError(e.message || 'error'))
      .finally(() => setBusy(false));
  };

  const activate = (c) => act(() => api.raw(`/ai/llm/connections/${c.id}/activate`, { method: 'POST' }));
  const remove = async (c) => {
    if (await window.uiConfirm({ title: t('llmcfg.delConfirm'), danger: true })) {
      act(() => api.raw(`/ai/llm/connections/${c.id}`, { method: 'DELETE' }));
    }
  };
  const save = () => act(() => {
    const body = toPayload(form.data);
    return form.mode === 'edit'
      ? api.raw(`/ai/llm/connections/${form.id}`, { method: 'PATCH', body })
      : api.raw('/ai/llm/connections', { method: 'POST', body });
  });
  const bindRole = (role, cid) =>
    act(() => api.raw(`/ai/llm/roles/${role}`, { method: 'PUT', body: { connection_id: cid || null } }));

  // Test connection — probe the SAVED endpoint + stored key server-side (the key never leaves the
  // server). The result is a sanitized {ok, category, status}; the pill localizes it by category.
  const testConn = (c) => {
    setTests((t) => ({ ...t, [c.id]: { loading: true } }));
    api.raw(`/ai/llm/connections/${c.id}/test`, { method: 'POST' })
      .then((r) => setTests((t) => ({ ...t, [c.id]: { ...r, loading: false } })))
      .catch((e) => setTests((t) => ({ ...t, [c.id]: { ok: false, category: 'error', detail: e.message, loading: false } })));
  };
  const testLabel = (r) => t('llmcfg.test.cat.' + (r.category || 'error')) + (r.status ? ` · HTTP ${r.status}` : '');

  const setF = (k) => (e) => setForm((f) => ({ ...f, data: { ...f.data, [k]: e.target.value } }));
  const fields = form ? providerFields(form.data.provider) : null;

  const columns = [
    { key: 'name', header: t('llmcfg.name'), render: (c) => (
      <span>
        <span>{c.name}</span>
        {c.is_active && <span className="badge on" style={{ marginLeft: 8 }} data-no-lex>{t('llmcfg.active')}</span>}
      </span>
    ) },
    { key: 'provider', header: t('llmcfg.provider'), className: 'prov-cell', render: (c) => {
      // Provider labels vary in length ("Anthropic" vs "กำหนดเอง (OpenAI-compatible)"); truncate to
      // one line with … so every row is the same height, and reveal the full name on hover/focus.
      const label = t('llmcfg.provider.' + c.provider);
      return <Tooltip label={label} className="prov-tip"><span data-no-lex>{label}</span></Tooltip>;
    } },
    { key: 'model', header: t('llmcfg.model'), render: (c) => <span className="mono faint llm-model" data-no-lex>{c.model || '—'}</span> },
    { key: 'key', header: t('llmcfg.apiKey'), render: (c) => (
      <span className={`badge ${c.api_key_set ? 'on' : ''}`} data-no-lex>
        {c.api_key_set ? t('llmcfg.keyStored') : t('llmcfg.keyNone')}
      </span>
    ) },
    { key: 'test', header: t('llmcfg.test.col'), render: (c) => {
      const r = tests[c.id];
      if (r && !r.loading) {
        // the pill IS the re-test control — click / Enter / Space re-probes (a fixed key/endpoint
        // must be verifiable without a full reload). Tooltip localizes the last result by category.
        const retest = () => { if (!busy) testConn(c); };
        return (
          <Tooltip label={testLabel(r)}>
            <span className={`badge ${r.ok ? 'on' : 'warn'} llm-test-pill`} data-no-lex tabIndex={0}
              role="button" aria-label={`${testLabel(r)} — ${t('llmcfg.test.btn')}`} aria-disabled={busy}
              onClick={retest} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); retest(); } }}>
              <span className="dot" />{r.ok ? t('llmcfg.test.ready') : t('llmcfg.test.notReady')}
            </span>
          </Tooltip>
        );
      }
      return <Button size="sm" icon="refresh" label={t('llmcfg.test.btn')} loading={!!(r && r.loading)}
        loadingLabel={t('llmcfg.test.testing')} disabled={busy} onClick={() => testConn(c)}>{t('llmcfg.test.btn')}</Button>;
    } },
    { key: 'act', header: '', render: (c) => (
      <span className="uc-act" onClick={(e) => e.stopPropagation()}>
        {!c.is_active && <Button size="sm" disabled={busy} onClick={() => activate(c)}>{t('llmcfg.activate')}</Button>}
        <Button size="sm" icon="edit" label={t('llmcfg.edit')} disabled={busy}
          onClick={() => setForm({ mode: 'edit', id: c.id, data: { name: c.name, provider: c.provider, model: c.model || '', base_url: c.base_url || '', api_key: '' } })} />
        <Button size="sm" kind="danger" icon="delete" label={t('llmcfg.del')} disabled={busy} onClick={() => remove(c)} />
      </span>
    ) },
  ];

  return (
    <div className="content-pad">
      <PageHead title={t('llmcfg.title')} desc={t('llmcfg.hint')}
        actions={<Button kind="gold" icon="add" disabled={busy}
          onClick={() => setForm({ mode: 'create', data: { ...EMPTY_FORM } })}>{t('llmcfg.add')}</Button>} />

      {error && <div className="badge warn" data-no-lex>{t('llmcfg.err')}: {error}</div>}

      <Panel title={t('llmcfg.title')}>
        {conns === null ? <p className="muted">{t('llmcfg.loading')}</p>
          : conns.length === 0 ? <Empty title={t('llmcfg.empty')} />
          : <div className="llm-conn-table"><Table columns={columns} rows={conns} /></div>}
      </Panel>

      <Panel title={t('llmcfg.roles.title')}>
        <p className="muted">{t('llmcfg.roles.hint')}</p>
        {roles.map((r) => {
          // Backend reports which plugin consumes the role + whether it's installed; a role whose
          // module is absent can't be bound (disabled + red note) — offering it misleads the operator.
          const missing = r.available === false;
          return (
            <Field key={r.role} label={t('llmcfg.role.' + r.role)} hint={t('llmcfg.role.' + r.role + '.desc')}
              error={missing ? `${t('llmcfg.roles.notInstalled')} (${r.plugin})` : undefined}>
              <Select block value={r.connection_id || ''} disabled={busy || missing}
                options={[{ value: '', label: t('llmcfg.roles.default') },
                  ...(conns || []).map((c) => ({ value: c.id, label: c.name }))]}
                onChange={(v) => bindRole(r.role, v)} />
            </Field>
          );
        })}
      </Panel>

      {form && (
        <Modal open onClose={() => setForm(null)}
          title={form.mode === 'edit' ? t('llmcfg.editTitle') : t('llmcfg.addTitle')}
          showClose closeLabel={t('common.close')}
          footer={<>
            <Button disabled={busy} onClick={() => setForm(null)}>{t('llmcfg.cancel')}</Button>
            <Button kind="gold" disabled={busy || !canSave(form.data, form.mode)} onClick={save}>{t('llmcfg.save')}</Button>
          </>}>
          <Field id="llm-name" label={t('llmcfg.name')} placeholder={t('llmcfg.namePh')}
            value={form.data.name} onChange={setF('name')} />
          <Field label={t('llmcfg.provider')}>
            <Select block value={form.data.provider}
              options={providerOptions(form.data.provider).map((p) => ({ value: p, label: t('llmcfg.provider.' + p) }))}
              onChange={(v) => setForm((f) => ({ ...f, data: { ...f.data, provider: v } }))} />
          </Field>
          <Field id="llm-model" label={t('llmcfg.model')} value={form.data.model} onChange={setF('model')} />
          <Field id="llm-base-url"
            label={fields.baseUrl === 'required' ? t('llmcfg.endpointCustom') : t('llmcfg.endpoint')}
            hint={fields.baseUrl === 'required' ? t('llmcfg.endpointCustomHint') : undefined}
            placeholder="http://localhost:1234/v1"
            value={form.data.base_url} onChange={setF('base_url')} />
          {fields.apiKey !== 'hidden' && (
            <Field id="llm-api-key" label={t('llmcfg.apiKey')} type="password"
              hint={form.mode === 'edit' ? t('llmcfg.apiKeyKeep')
                : fields.apiKey === 'optional' ? t('llmcfg.apiKeyOptional') : undefined}
              value={form.data.api_key} onChange={setF('api_key')} autoComplete="new-password" />
          )}
          {fields.apiKey === 'hidden' && <p className="muted">{t('llmcfg.apiKeyLocal')}</p>}
        </Modal>
      )}
    </div>
  );
}
