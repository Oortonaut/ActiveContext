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
            const response = await fetch(`/api/sessions/${this.sessionId}/message-history`);
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
            // Handle pong response from server (plain string, not JSON)
            if (event.data === 'pong') {
                return;
            }
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (err) {
                console.error('Failed to parse WebSocket message:', err, event.data);
            }
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
        } else if (data.type === 'expansion_changed') {
            // Expansion change confirmed - context will be reloaded via node_changed broadcast
            console.log(`Expansion changed: ${data.node_id} ${data.old_expansion} -> ${data.new_expansion}`);
        } else if (data.type === 'hidden_changed') {
            // Hidden flag changed - context will be reloaded via node_changed broadcast
            console.log(`Hidden changed: ${data.node_id} ${data.old_hidden} -> ${data.new_hidden}`);
        } else if (data.type === 'error') {
            console.error('Server error:', data.message);
        }
    }

    // Node expansion control methods
    setNodeExpansion(nodeId, newExpansion) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected');
            return;
        }

        this.ws.send(JSON.stringify({
            type: 'set_expansion',
            node_id: nodeId,
            expansion: newExpansion
        }));
    }

    toggleNodeHidden(nodeId, currentHidden) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected');
            return;
        }

        this.ws.send(JSON.stringify({
            type: 'set_hidden',
            node_id: nodeId,
            hidden: !currentHidden
        }));
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
                this.loadRendered();
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
        } else {
            acpInfo.classList.add('hidden');
            directInfo.classList.remove('hidden');
        }
    }

    renderCapabilitiesTo(capabilities, listId, countId) {
        const listEl = document.getElementById(listId);
        const countEl = countId ? document.getElementById(countId) : null;
        
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

        // Render client capabilities in features card
        const capsSection = document.getElementById('features-caps-section');
        if (data.client && data.client.capabilities) {
            if (capsSection) capsSection.classList.remove('hidden');
            this.renderCapabilitiesTo(
                data.client.capabilities,
                'features-capabilities-list',
                'features-caps-count'
            );
        } else if (capsSection) {
            capsSection.classList.add('hidden');
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
        
        // Get the container for dynamic sections
        const container = document.getElementById('context-card').querySelector('.card-content');
        container.innerHTML = '';
        
        // Define display order (known types first, then alphabetical for others)
        const knownOrder = ['view', 'group', 'markdown', 'topic', 'artifact', 'shell', 'mcp_manager', 'session', 'message'];
        const nodesByType = data.nodes_by_type || {};
        const types = Object.keys(nodesByType);
        
        // Sort: known types in order, then unknown types alphabetically
        types.sort((a, b) => {
            const aIdx = knownOrder.indexOf(a);
            const bIdx = knownOrder.indexOf(b);
            if (aIdx >= 0 && bIdx >= 0) return aIdx - bIdx;
            if (aIdx >= 0) return -1;
            if (bIdx >= 0) return 1;
            return a.localeCompare(b);
        });
        
        // Render each type dynamically
        types.forEach(type => {
            const items = nodesByType[type];
            const section = document.createElement('div');
            section.className = 'context-section';
            
            // Collapse less common types by default
            const commonTypes = ['view', 'group', 'markdown', 'trace'];
            if (!commonTypes.includes(type)) {
                section.classList.add('collapsed');
            }
            
            const displayName = this.formatTypeName(type);
            section.innerHTML = `
                <h3>${displayName} <span class="badge small">${items.length}</span></h3>
                <ul class="context-list"></ul>
            `;
            
            const list = section.querySelector('ul');
            this.renderContextListItems(list, items, type);
            container.appendChild(section);
        });
        
        // Show empty state if no nodes
        if (types.length === 0) {
            container.innerHTML = '<div class="context-empty">No context objects</div>';
        }
    }
    
    formatTypeName(type) {
        // Convert snake_case to Title Case
        return type.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    renderContextListItems(list, items, type) {
        list.innerHTML = '';

        items.forEach(item => {
            const li = document.createElement('li');
            const nodeId = item.id || item.node_id;
            const currentExpansion = item.expansion || 'all';
            const isHidden = item.hidden || false;

            // Add hidden class for styling
            if (isHidden) {
                li.classList.add('node-hidden');
            }

            // Get details based on type
            let details = this.getNodeDetails(item, type);

            // Show parent indicator if node has parents
            const hasParent = item.parent_ids && item.parent_ids.length > 0;
            const parentInfo = hasParent ? `<span class="node-parent">⤴ ${item.parent_ids[0]}</span>` : '';

            // Eye icon: open eye when visible, closed eye when hidden
            const eyeIcon = isHidden ? '👁‍🗨' : '👁';

            li.innerHTML = `
                <span class="node-id">${nodeId}</span>
                <span class="node-path ${isHidden ? 'strikethrough' : ''}">${details}</span>
                ${parentInfo}
                <div class="node-controls">
                    <button class="node-toggle-btn" data-node-id="${nodeId}" data-hidden="${isHidden}" title="Toggle visibility">
                        ${eyeIcon}
                    </button>
                    <select class="node-expansion-select" data-node-id="${nodeId}" title="Change expansion level" ${isHidden ? 'disabled' : ''}>
                        <option value="header" ${currentExpansion === 'header' ? 'selected' : ''}>Header</option>
                        <option value="content" ${currentExpansion === 'content' ? 'selected' : ''}>Content</option>
                        <option value="index" ${currentExpansion === 'index' ? 'selected' : ''}>Index</option>
                        <option value="all" ${currentExpansion === 'all' ? 'selected' : ''}>All</option>
                    </select>
                </div>
            `;

            // Add event listeners
            const toggleBtn = li.querySelector('.node-toggle-btn');
            toggleBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleNodeHidden(nodeId, isHidden);
            });

            const expansionSelect = li.querySelector('.node-expansion-select');
            expansionSelect.addEventListener('change', (e) => {
                e.stopPropagation();
                this.setNodeExpansion(nodeId, e.target.value);
            });

            list.appendChild(li);
        });
    }
    
    getNodeDetails(item, type) {
        // Type-specific detail extraction
        switch (type) {
            case 'view':
                return item.path || '';
            case 'group':
                const childCount = item.children_ids ? item.children_ids.length : 0;
                return `${childCount} children`;
            case 'topic':
                return item.title || '';
            case 'artifact':
                return item.artifact_type || '';
            case 'message':
                return item.role || '';
            case 'markdown':
                return item.path || item.title || '';
            case 'shell':
                return item.command || '';
            case 'session':
                return item.cwd || '';
            case 'mcp_manager':
                return `${item.server_count || 0} servers`;
            default:
                // Try common fields
                return item.path || item.title || item.name || '';
        }
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

        // No budget - just show total tokens used
        progress.style.width = '0%';
        usage.textContent = `${data.total_used.toLocaleString()} tokens`;

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
            if (msg.originator) {
                if (msg.originator === 'user') {
                    label = 'User';
                } else if (msg.originator === 'agent') {
                    label = 'Agent';
                } else if (msg.originator.startsWith('tool:')) {
                    label = msg.originator.substring(5);
                } else if (msg.originator.startsWith('child:')) {
                    label = `Child: ${msg.originator.substring(6)}`;
                } else {
                    label = msg.originator;
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
        console.log('renderRenderedContext called with:', data);
        
        const tokensEl = document.getElementById('rendered-tokens');
        if (tokensEl) {
            const totalTokens = data.total_tokens || 0;
            tokensEl.textContent = `${totalTokens.toLocaleString()} tokens`;
        }

        const contentEl = document.getElementById('rendered-content');
        if (contentEl) {
            try {
            // Check for actual content (not just truthy - empty string is valid but shows as empty)
            if (data.rendered !== undefined && data.rendered !== null && data.rendered.length > 0) {
                contentEl.innerHTML = `<pre class="rendered-pre">${this.escapeHtml(data.rendered)}</pre>`;
            } else if (data.sections && data.sections.length > 0) {
                // If we have sections but no rendered content, show that
                contentEl.innerHTML = '<div class="rendered-empty">Projection has sections but no rendered text</div>';
            } else {
                contentEl.innerHTML = '<div class="rendered-empty">No projection rendered yet (no messages or context)</div>';
            }
            } catch (err) {
                console.error('Error rendering content:', err);
                contentEl.innerHTML = '<div class="rendered-empty">Error: ' + err.message + '</div>';
            }
        }

        // Render sections breakdown
        const sectionsEl = document.getElementById('rendered-sections');
        console.log('sectionsEl:', sectionsEl, 'data.sections:', data.sections);
        if (sectionsEl) {
            sectionsEl.innerHTML = '';
            
            if (!data.sections || !Array.isArray(data.sections) || data.sections.length === 0) {
                console.log('No sections, showing empty state');
                sectionsEl.innerHTML = '<div class="rendered-empty">No sections to display</div>';
            } else {
                console.log('Rendering', data.sections.length, 'sections');
                try {
                data.sections.forEach((section, index) => {
                const sectionItem = document.createElement('div');
                const nodeId = section.source_id;
                const currentExpansion = section.expansion || 'all';
                const isHidden = section.hidden || false;

                sectionItem.className = 'section-item';
                if (isHidden) {
                    sectionItem.classList.add('section-hidden');
                }

                // Eye icon for toggle
                const eyeIcon = isHidden ? '👁‍🗨' : '👁';

                sectionItem.innerHTML = `
                    <div class="section-header">
                        <span class="section-type">${section.type}</span>
                        <span class="section-source ${isHidden ? 'strikethrough' : ''}">${nodeId}</span>
                        <span class="section-tokens">${section.tokens_used} tokens</span>
                        <div class="section-controls">
                            <button class="section-toggle-btn" data-node-id="${nodeId}" data-hidden="${isHidden}" title="Toggle visibility">
                                ${eyeIcon}
                            </button>
                            <select class="section-expansion-select" data-node-id="${nodeId}" title="Change expansion level" ${isHidden ? 'disabled' : ''}>
                                <option value="header" ${currentExpansion === 'header' ? 'selected' : ''}>Header</option>
                                <option value="content" ${currentExpansion === 'content' ? 'selected' : ''}>Content</option>
                                <option value="index" ${currentExpansion === 'index' ? 'selected' : ''}>Index</option>
                                <option value="all" ${currentExpansion === 'all' ? 'selected' : ''}>All</option>
                            </select>
                        </div>
                    </div>
                    <pre class="section-content">${this.escapeHtml(section.content || '')}</pre>
                `;

                // Add event listeners
                const toggleBtn = sectionItem.querySelector('.section-toggle-btn');
                toggleBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.toggleNodeHidden(nodeId, isHidden);
                });

                const expansionSelect = sectionItem.querySelector('.section-expansion-select');
                expansionSelect.addEventListener('change', (e) => {
                    e.stopPropagation();
                    this.setNodeExpansion(nodeId, e.target.value);
                });

                sectionsEl.appendChild(sectionItem);
            });
                } catch (err) {
                    console.error('Error rendering sections:', err);
                    sectionsEl.innerHTML = '<div class="rendered-empty">Error rendering sections: ' + err.message + '</div>';
                }
            }
        }

        // Store data for copy button
        this._renderedData = data;
    }

    setupRenderedControls() {
        // Raw mode checkbox - controls whether to show raw text or sections
        const rawCheckbox = document.getElementById('raw-mode-checkbox');
        const contentEl = document.getElementById('rendered-content');
        const sectionsEl = document.getElementById('rendered-sections');

        // Set initial state: non-Raw mode (sections visible, raw content hidden)
        if (contentEl && sectionsEl) {
            contentEl.classList.add('hidden');
            sectionsEl.classList.remove('hidden');
        }

        if (rawCheckbox) {
            rawCheckbox.addEventListener('change', () => {
                if (rawCheckbox.checked) {
                    // Raw mode: show raw content, hide sections
                    contentEl.classList.remove('hidden');
                    sectionsEl.classList.add('hidden');
                } else {
                    // Non-Raw mode: show sections, hide raw content
                    contentEl.classList.add('hidden');
                    sectionsEl.classList.remove('hidden');
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
            total_used: 0,
            sections: []
        });
        this.renderConversation({ messages: [], count: 0 });
        this.renderRenderedContext({
            rendered: '',
            total_tokens: 0,
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
