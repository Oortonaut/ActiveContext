/**
 * ActiveContext Dashboard Client
 * Handles REST API calls and WebSocket connections for real-time updates.
 */

class DashboardClient {
    constructor() {
        this.sessionId = null;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000;
        this.pingInterval = null;
        this.llmRefreshInterval = null;
    }

    async init() {
        // Load initial data
        await this.loadClient();
        await this.loadLLMStatus();
        await this.loadSessions();

        // Set up session selector
        const select = document.getElementById('session-select');
        select.addEventListener('change', (e) => {
            this.selectSession(e.target.value);
        });

        // Set up rendered context controls
        this.setupRenderedControls();

        // Set up copy session ID button
        this.setupCopySessionId();

        // Start uptime counter
        this.startUptimeCounter();

        // Periodically refresh LLM status (every 5 seconds)
        this.llmRefreshInterval = setInterval(() => {
            this.loadLLMStatus();
        }, 5000);
    }

    async loadLLMStatus() {
        try {
            const response = await fetch('/api/llm');
            const data = await response.json();
            this.renderLLMStatus(data);
        } catch (error) {
            console.error('Failed to load LLM status:', error);
        }
    }

    async loadSessions() {
        try {
            const response = await fetch('/api/sessions');
            const sessions = await response.json();
            this.renderSessionSelector(sessions);

            // Auto-select first session
            if (sessions.length > 0) {
                this.selectSession(sessions[0].session_id);
            }
        } catch (error) {
            console.error('Failed to load sessions:', error);
        }
    }

    async selectSession(sessionId) {
        if (!sessionId) {
            this.sessionId = null;
            this.disconnect();
            this.clearSessionData();
            return;
        }

        this.sessionId = sessionId;

        // Update selector
        const select = document.getElementById('session-select');
        select.value = sessionId;

        // Load session data
        await Promise.all([
            this.loadFeatures(),
            this.loadContext(),
            this.loadTimeline(),
            this.loadProjection(),
            this.loadConversation(),
            this.loadRendered()
        ]);

        // Connect WebSocket
        this.connect();
    }

    async loadContext() {
        if (!this.sessionId) return;

        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/context`);
            const data = await response.json();
            this.renderContext(data);
        } catch (error) {
            console.error('Failed to load context:', error);
        }
    }

    async loadTimeline() {
        if (!this.sessionId) return;

        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/timeline`);
            const data = await response.json();
            this.renderTimeline(data);
        } catch (error) {
            console.error('Failed to load timeline:', error);
        }
    }

    async loadProjection() {
        if (!this.sessionId) return;

        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/projection`);
            const data = await response.json();
            this.renderProjection(data);
        } catch (error) {
            console.error('Failed to load projection:', error);
        }
    }

    async loadConversation() {
        if (!this.sessionId) return;

        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/conversation`);
            const data = await response.json();
            this.renderConversation(data);
        } catch (error) {
            console.error('Failed to load conversation:', error);
        }
    }

    async loadRendered() {
        if (!this.sessionId) return;

        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/rendered`);
            const data = await response.json();
            this.renderRenderedContext(data);
        } catch (error) {
            console.error('Failed to load rendered context:', error);
        }
    }

    async loadClient() {
        try {
            const response = await fetch('/api/client');
            const data = await response.json();
            this._clientData = data;
            this.renderClient(data);
        } catch (error) {
            console.error('Failed to load client info:', error);
        }
    }

    async loadFeatures() {
        if (!this.sessionId) return;

        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/features`);
            const data = await response.json();
            this.renderFeatures(data);
        } catch (error) {
            console.error('Failed to load session features:', error);
        }
    }

    // WebSocket connection
    connect() {
        if (!this.sessionId) return;

        this.disconnect();
        this.setConnectionStatus('connecting');

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.setConnectionStatus('connected');
            this.reconnectAttempts = 0;
            this.startPing();
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.setConnectionStatus('disconnected');
            this.stopPing();
            this.scheduleReconnect();
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    disconnect() {
        this.stopPing();
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Max reconnect attempts reached');
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1);

        setTimeout(() => {
            if (this.sessionId) {
                console.log(`Reconnecting (attempt ${this.reconnectAttempts})...`);
                this.connect();
            }
        }, delay);
    }

    startPing() {
        this.pingInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send('ping');
            }
        }, 30000);
    }

    stopPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    handleMessage(data) {
        if (data.type === 'init') {
            // Initial state from WebSocket
            if (data.client) {
                this._clientData = data.client;
                this.renderClient(data.client);
            }
            if (data.features) {
                this.renderFeatures(data.features);
            }
            this.renderContext(data.context);
            this.renderTimeline(data.timeline);
            this.renderProjection(data.projection);
            if (data.conversation) {
                this.renderConversation(data.conversation);
            }
            if (data.rendered) {
                this.renderRenderedContext(data.rendered);
            }
        } else if (data.type === 'update') {
            this.handleUpdate(data);
        }
    }

    handleUpdate(data) {
        switch (data.kind) {
            case 'statement_executed':
            case 'statement_executing':
                this.loadTimeline();
                break;
            case 'node_changed':
                this.loadContext();
                this.loadConversation();
                break;
            case 'projection_ready':
                this.loadProjection();
                this.loadRendered();
                break;
            case 'message_added':
                this.loadConversation();
                this.loadRendered();
                break;
        }
    }

    // Rendering methods
    renderLLMStatus(data) {
        const modelEl = document.getElementById('current-model');
        const providerEl = document.getElementById('current-provider');
        const availableEl = document.getElementById('available-providers');

        if (data.current_model) {
            modelEl.textContent = data.current_model;

            // Find provider for current model
            const model = data.available_models.find(m => m.model_id === data.current_model);
            providerEl.textContent = model ? model.provider : '-';
        } else {
            modelEl.textContent = 'Not configured';
            providerEl.textContent = '-';
        }

        availableEl.textContent = data.available_providers.join(', ') || 'None';
    }

    renderClient(data) {
        const transportEl = document.getElementById('transport-type');
        const acpInfo = document.getElementById('acp-info');
        const directInfo = document.getElementById('direct-info');

        if (!transportEl) return;

        transportEl.textContent = data.transport.type.toUpperCase();
        transportEl.className = `stat-value badge ${data.transport.is_acp ? 'acp' : 'direct'}`;

        if (data.transport.is_acp && data.client) {
            acpInfo.classList.remove('hidden');
            directInfo.classList.add('hidden');

            const displayName = data.client.title || data.client.name;
            document.getElementById('client-name').textContent = displayName;
            document.getElementById('client-version').textContent = data.client.version;
            document.getElementById('protocol-version').textContent = `v${data.protocol_version || '?'}`;

            // Render capabilities dynamically
            this.renderCapabilities(data.client.capabilities);
        } else {
            acpInfo.classList.add('hidden');
            directInfo.classList.remove('hidden');
        }
    }

    renderCapabilities(capabilities) {
        const listEl = document.getElementById('capabilities-list');
        const countEl = document.getElementById('caps-count');
        
        if (!listEl) return;
        listEl.innerHTML = '';

        if (!capabilities || !Array.isArray(capabilities) || capabilities.length === 0) {
            listEl.innerHTML = '<div class="muted">No capabilities reported</div>';
            if (countEl) countEl.textContent = '0';
            return;
        }

        if (countEl) countEl.textContent = capabilities.length;

        capabilities.forEach(cap => {
            const row = document.createElement('div');
            row.className = 'capability-row';
            row.title = cap.description || cap.name;

            const icon = document.createElement('span');
            icon.className = `capability-icon ${cap.enabled ? 'enabled' : 'disabled'}`;
            icon.textContent = cap.enabled ? '✓' : '✗';

            const label = document.createElement('span');
            label.className = 'capability-label';
            label.textContent = cap.label || cap.name;

            // Show extension indicator for _meta capabilities
            if (cap.name.includes('_meta')) {
                const extBadge = document.createElement('span');
                extBadge.className = 'badge small ext';
                extBadge.textContent = 'ext';
                label.appendChild(extBadge);
            }

            // Show value if present (for non-boolean extensions)
            if (cap.value !== undefined && cap.value !== null) {
                const valueEl = document.createElement('span');
                valueEl.className = 'capability-value';
                valueEl.textContent = typeof cap.value === 'object' 
                    ? JSON.stringify(cap.value) 
                    : String(cap.value);
                row.appendChild(icon);
                row.appendChild(label);
                row.appendChild(valueEl);
            } else {
                row.appendChild(icon);
                row.appendChild(label);
            }

            listEl.appendChild(row);
        });
    }

    renderFeatures(data) {
        const modeEl = document.getElementById('session-mode');
        const modelEl = document.getElementById('session-model');
        const cwdEl = document.getElementById('session-cwd');

        if (modeEl) {
            modeEl.textContent = data.mode || 'default';
        }
        if (modelEl) {
            modelEl.textContent = data.model || 'Not configured';
        }
        if (cwdEl) {
            cwdEl.textContent = data.cwd || '-';
            cwdEl.title = data.cwd || '';  // Full path on hover
        }
    }

    renderSessionSelector(sessions) {
        const select = document.getElementById('session-select');
        select.innerHTML = '';

        if (sessions.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No sessions';
            select.appendChild(option);
            return;
        }

        sessions.forEach(session => {
            const option = document.createElement('option');
            option.value = session.session_id;
            option.textContent = `${session.session_id} (${session.cwd})`;
            select.appendChild(option);
        });
    }

    renderContext(data) {
        document.getElementById('context-count').textContent = data.total;
        document.getElementById('views-count').textContent = data.views.length;
        document.getElementById('groups-count').textContent = data.groups.length;
        document.getElementById('topics-count').textContent = data.topics.length;
        document.getElementById('artifacts-count').textContent = data.artifacts.length;

        this.renderContextList('views-list', data.views, 'view');
        this.renderContextList('groups-list', data.groups, 'group');
        this.renderContextList('topics-list', data.topics, 'topic');
        this.renderContextList('artifacts-list', data.artifacts, 'artifact');

        // Render messages if element exists
        const messagesCount = document.getElementById('messages-count');
        if (messagesCount && data.messages) {
            messagesCount.textContent = data.messages.length;
            this.renderContextList('messages-list', data.messages, 'message');
        }
    }

    renderContextList(elementId, items, type) {
        const list = document.getElementById(elementId);
        if (!list) return;

        list.innerHTML = '';

        items.forEach(item => {
            const li = document.createElement('li');

            let details = '';
            if (type === 'view' && item.path) {
                details = item.path;
            } else if (type === 'group') {
                const childCount = item.children_ids ? item.children_ids.length : 0;
                details = `${childCount} children`;
            } else if (type === 'topic') {
                details = item.title || '';
            } else if (type === 'artifact') {
                details = item.artifact_type || '';
            } else if (type === 'message') {
                details = item.role || '';
            }

            // Show parent indicator if node has parents
            const hasParent = item.parent_ids && item.parent_ids.length > 0;
            const parentInfo = hasParent ? `<span class="node-parent">⤴ ${item.parent_ids[0]}</span>` : '';

            li.innerHTML = `
                <span class="node-id">${item.id || item.node_id}</span>
                <span class="node-path">${details}</span>
                ${parentInfo}
                <span class="node-state">${item.state || 'details'}</span>
            `;
            list.appendChild(li);
        });
    }

    renderTimeline(data) {
        document.getElementById('statement-count').textContent = data.count;

        const list = document.getElementById('timeline-list');

        if (data.statements.length === 0) {
            list.innerHTML = '<div class="timeline-empty">No statements executed yet</div>';
            return;
        }

        list.innerHTML = '';

        // Show most recent first
        const statements = [...data.statements].reverse();

        statements.forEach(stmt => {
            const item = document.createElement('div');
            item.className = 'timeline-item';

            const statusClass = stmt.status === 'ok' ? 'ok' :
                               stmt.status === 'error' ? 'error' : 'pending';
            const statusIcon = stmt.status === 'ok' ? '✓' :
                              stmt.status === 'error' ? '✗' : '…';

            const time = new Date(stmt.timestamp * 1000).toLocaleTimeString();

            item.innerHTML = `
                <div class="timeline-status ${statusClass}">${statusIcon}</div>
                <div class="timeline-content">
                    <code class="timeline-source">${this.escapeHtml(stmt.source)}</code>
                    <div class="timeline-meta">
                        <span>#${stmt.index}</span>
                        <span>${time}</span>
                        ${stmt.duration_ms > 0 ? `<span>${stmt.duration_ms.toFixed(1)}ms</span>` : ''}
                    </div>
                </div>
            `;
            list.appendChild(item);
        });
    }

    renderProjection(data) {
        const progress = document.querySelector('#token-progress .progress-fill');
        const usage = document.getElementById('token-usage');
        const utilization = (data.utilization * 100).toFixed(1);

        progress.style.width = `${utilization}%`;
        usage.textContent = `${data.total_used.toLocaleString()} / ${data.total_budget.toLocaleString()} (${utilization}%)`;

        // Calculate tokens by section type
        let convTokens = 0, viewsTokens = 0, groupsTokens = 0;

        data.sections.forEach(section => {
            if (section.type === 'conversation') {
                convTokens += section.tokens_used;
            } else if (section.type === 'view') {
                viewsTokens += section.tokens_used;
            } else if (section.type === 'group') {
                groupsTokens += section.tokens_used;
            }
        });

        document.getElementById('conv-tokens').textContent = convTokens.toLocaleString();
        document.getElementById('views-tokens').textContent = viewsTokens.toLocaleString();
        document.getElementById('groups-tokens').textContent = groupsTokens.toLocaleString();
    }

    renderConversation(data) {
        const countEl = document.getElementById('conversation-count');
        if (countEl) {
            countEl.textContent = data.count;
        }

        const list = document.getElementById('conversation-list');
        if (!list) return;

        if (data.messages.length === 0) {
            list.innerHTML = '<div class="conversation-empty">No messages yet</div>';
            return;
        }

        list.innerHTML = '';

        data.messages.forEach(msg => {
            const item = document.createElement('div');
            item.className = `conversation-item role-${msg.role}`;

            // Determine display label
            let label = msg.role;
            if (msg.actor) {
                if (msg.actor === 'user') {
                    label = 'User';
                } else if (msg.actor === 'agent') {
                    label = 'Agent';
                } else if (msg.actor.startsWith('tool:')) {
                    label = msg.actor.substring(5);
                } else if (msg.actor.startsWith('child:')) {
                    label = `Child: ${msg.actor.substring(6)}`;
                } else {
                    label = msg.actor;
                }
            } else if (msg.role === 'tool_call') {
                label = `Tool Call: ${msg.tool_name || 'unknown'}`;
            } else if (msg.role === 'tool_result') {
                label = `Result: ${msg.tool_name || ''}`;
            }

            // Format content (truncate if too long)
            let content = msg.content || '';
            const maxLen = 500;
            const isTruncated = content.length > maxLen;
            const displayContent = isTruncated ? content.substring(0, maxLen) + '...' : content;

            item.innerHTML = `
                <div class="message-header">
                    <span class="message-role">${this.escapeHtml(label)}</span>
                    <span class="message-id">[${msg.id}]</span>
                </div>
                <div class="message-content">${this.escapeHtml(displayContent)}</div>
                ${isTruncated ? '<div class="message-truncated">(truncated)</div>' : ''}
            `;

            // Add click to expand
            if (isTruncated) {
                item.addEventListener('click', () => {
                    const contentEl = item.querySelector('.message-content');
                    const truncatedEl = item.querySelector('.message-truncated');
                    if (contentEl.textContent === displayContent) {
                        contentEl.textContent = content;
                        truncatedEl.textContent = '(click to collapse)';
                    } else {
                        contentEl.textContent = displayContent;
                        truncatedEl.textContent = '(truncated)';
                    }
                });
                item.style.cursor = 'pointer';
            }

            list.appendChild(item);
        });
    }

    renderRenderedContext(data) {
        const tokensEl = document.getElementById('rendered-tokens');
        if (tokensEl) {
            const totalTokens = data.total_tokens || 0;
            const tokenBudget = data.token_budget || 0;
            tokensEl.textContent = `${totalTokens.toLocaleString()} / ${tokenBudget.toLocaleString()} tokens`;
        }

        const contentEl = document.getElementById('rendered-content');
        if (contentEl) {
            // Check for actual content (not just truthy - empty string is valid but shows as empty)
            if (data.rendered !== undefined && data.rendered !== null && data.rendered.length > 0) {
                contentEl.innerHTML = `<pre class="rendered-pre">${this.escapeHtml(data.rendered)}</pre>`;
            } else if (data.sections && data.sections.length > 0) {
                // If we have sections but no rendered content, show that
                contentEl.innerHTML = '<div class="rendered-empty">Projection has sections but no rendered text</div>';
            } else {
                contentEl.innerHTML = '<div class="rendered-empty">No projection rendered yet (no messages or context)</div>';
            }
        }

        // Render sections breakdown
        const sectionsEl = document.getElementById('rendered-sections');
        if (sectionsEl && data.sections) {
            sectionsEl.innerHTML = '';
            data.sections.forEach((section, index) => {
                const sectionItem = document.createElement('div');
                sectionItem.className = 'section-item';
                sectionItem.innerHTML = `
                    <div class="section-header">
                        <span class="section-type">${section.type}</span>
                        <span class="section-source">${section.source_id}</span>
                        <span class="section-tokens">${section.tokens_used} tokens</span>
                        <span class="section-state">${section.state}</span>
                    </div>
                    <pre class="section-content">${this.escapeHtml(section.content || '')}</pre>
                `;
                sectionsEl.appendChild(sectionItem);
            });
        }

        // Store data for copy button
        this._renderedData = data;
    }

    setupRenderedControls() {
        // Toggle sections view
        const toggleBtn = document.getElementById('toggle-sections');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                const contentEl = document.getElementById('rendered-content');
                const sectionsEl = document.getElementById('rendered-sections');
                if (sectionsEl.classList.contains('hidden')) {
                    sectionsEl.classList.remove('hidden');
                    contentEl.classList.add('hidden');
                    toggleBtn.textContent = 'Show Full';
                } else {
                    sectionsEl.classList.add('hidden');
                    contentEl.classList.remove('hidden');
                    toggleBtn.textContent = 'Show Sections';
                }
            });
        }

        // Copy button
        const copyBtn = document.getElementById('copy-rendered');
        if (copyBtn) {
            copyBtn.addEventListener('click', async () => {
                if (this._renderedData && this._renderedData.rendered) {
                    try {
                        await navigator.clipboard.writeText(this._renderedData.rendered);
                        copyBtn.textContent = 'Copied!';
                        setTimeout(() => { copyBtn.textContent = 'Copy'; }, 2000);
                    } catch (err) {
                        console.error('Failed to copy:', err);
                    }
                }
            });
        }
    }

    setupCopySessionId() {
        const copyBtn = document.getElementById('copy-session-id');
        if (copyBtn) {
            copyBtn.addEventListener('click', async () => {
                if (this.sessionId) {
                    try {
                        await navigator.clipboard.writeText(this.sessionId);
                        copyBtn.textContent = '✓ Copied!';
                        setTimeout(() => { copyBtn.textContent = '📋 Copy ID'; }, 2000);
                    } catch (err) {
                        console.error('Failed to copy session ID:', err);
                    }
                }
            });
        }
    }

    clearSessionData() {
        this.renderFeatures({ mode: null, model: null, cwd: '-' });
        this.renderContext({ views: [], groups: [], topics: [], artifacts: [], messages: [], total: 0 });
        this.renderTimeline({ statements: [], count: 0 });
        this.renderProjection({
            total_budget: 0,
            total_used: 0,
            utilization: 0,
            sections: []
        });
        this.renderConversation({ messages: [], count: 0 });
        this.renderRenderedContext({
            rendered: '',
            total_tokens: 0,
            token_budget: 0,
            sections: [],
            section_count: 0
        });
    }

    setConnectionStatus(status) {
        const el = document.getElementById('connection-status');
        el.className = `status-indicator ${status}`;
        el.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    }

    startUptimeCounter() {
        const startTime = Date.now();
        const uptimeEl = document.getElementById('uptime');

        setInterval(() => {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const hours = Math.floor(elapsed / 3600);
            const minutes = Math.floor((elapsed % 3600) / 60);
            const seconds = elapsed % 60;

            if (hours > 0) {
                uptimeEl.textContent = `Uptime: ${hours}h ${minutes}m ${seconds}s`;
            } else if (minutes > 0) {
                uptimeEl.textContent = `Uptime: ${minutes}m ${seconds}s`;
            } else {
                uptimeEl.textContent = `Uptime: ${seconds}s`;
            }
        }, 1000);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const dashboard = new DashboardClient();
    dashboard.init();
});
