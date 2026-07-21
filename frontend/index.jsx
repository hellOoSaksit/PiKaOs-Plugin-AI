/* AI plugin — frontend descriptor. Contributes the admin LLM-config screen (connections CRUD +
   activate + role binding) through Core's plugin seam. Present only when this plugin is linked;
   a kernel-only Core ships none of it. The nav item is gated on llm.view: with auth installed
   only holders see it; in open/no-auth mode can() is allow-all so everyone does — enforcement
   stays server-side. `icon` names come from Core's design-system set (icons.jsx). */
import React from 'react';
import { LlmConfig } from './LlmConfig.jsx';

export default {
  id: 'ai',
  routes: [
    { id: 'llm-config',
      meta: { icon: 'ai', title: 'ตั้งค่า AI', en: 'AI Settings' },
      render: (ctx) => <LlmConfig ctx={ctx} /> },
  ],
  // Merges into Core's existing "ผู้ดูแลระบบ" group as a sibling of Settings (nesting under a
  // Core item isn't a seam capability — an admin can arrange it via the nav editor).
  nav: [
    { group: 'ผู้ดูแลระบบ',
      items: [{ id: 'llm-config', icon: 'ai', perm: 'llm.view' }] },
  ],
  // llm.* perms already ship in the backend manifest — declaring them here too would double
  // the RBAC catalog (same reason the auth descriptor's list is empty).
  permissions: [],
};
