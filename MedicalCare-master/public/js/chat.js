document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const resetBtn = document.getElementById('resetBtn');
    
    // Track if we're waiting for a response
    let isWaiting = false;
    
    // Generate a unique user ID and store it in localStorage
    let userId = localStorage.getItem('medicalCareUserId');
    if (!userId) {
        userId = generateUUID();
        localStorage.setItem('medicalCareUserId', userId);
    }
    
    // Initialize with stored thread ID or null
    let threadId = localStorage.getItem('medicalCareThreadId');
    
    // Helper function to generate UUID
    function generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
    
    // Helper function to format time
    function formatTime() {
        const now = new Date();
        return `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
    }
    
    // Add a message to the chat
    function addMessage(content, isUser = false) {
        if (!chatMessages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', isUser ? 'user' : 'assistant');
        
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        contentDiv.textContent = content;
        
        const timeDiv = document.createElement('div');
        timeDiv.classList.add('message-time');
        timeDiv.textContent = formatTime();
        
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeDiv);
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Show typing indicator
    function showTypingIndicator() {
        if (!chatMessages) return;
        
        const indicator = document.createElement('div');
        indicator.classList.add('typing-indicator');
        indicator.id = 'typingIndicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            indicator.appendChild(dot);
        }
        
        chatMessages.appendChild(indicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Hide typing indicator
    function hideTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    // Send message to server
    async function sendMessage(text) {
        if (!text.trim() || isWaiting || !chatMessages) return;
        
        isWaiting = true;
        if (sendBtn) sendBtn.disabled = true;
        
        // Add user message to chat
        addMessage(text, true);
        if (userInput) userInput.value = '';
        
        showTypingIndicator();
        
        try {
            const response = await fetch('http://localhost:5001/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_input: text,
                    user_id: userId,
                    thread_id: threadId
                })
            });
            
            const data = await response.json();
            
            // Hide typing indicator
            hideTypingIndicator();
            
            // Save thread ID
            if (data.thread_id) {
                threadId = data.thread_id;
                localStorage.setItem('medicalCareThreadId', threadId);
            }
            
            // Display responses
            if (data.responses && data.responses.length > 0) {
                data.responses.forEach(resp => {
                    addMessage(resp);
                });
            } else if (data.error) {
                addMessage("Xin lỗi, đã có lỗi xảy ra: " + data.error);
            }
        } catch (error) {
            console.error('Error:', error);
            hideTypingIndicator();
            addMessage("Xin lỗi, không thể kết nối đến máy chủ. Vui lòng thử lại sau.");
        }
        
        isWaiting = false;
        if (sendBtn) sendBtn.disabled = false;
    }
    
    // Reset the conversation
    async function resetConversation() {
        if (!chatMessages) return;
        
        try {
            const response = await fetch('http://localhost:5001/reset', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Clear all chat messages
                chatMessages.innerHTML = '';
                
                // Clear thread ID
                threadId = null;
                localStorage.removeItem('medicalCareThreadId');
                
                // Add back the initial greeting message
                const welcomeMessage = document.createElement('div');
                welcomeMessage.classList.add('message', 'assistant');
                
                const contentDiv = document.createElement('div');
                contentDiv.classList.add('message-content');
                contentDiv.textContent = "Xin chào! Tôi là trợ lý y tế MedicalCare. Tôi có thể giúp gì cho bạn hôm nay?";
                
                const timeDiv = document.createElement('div');
                timeDiv.classList.add('message-time');
                timeDiv.textContent = formatTime();
                
                welcomeMessage.appendChild(contentDiv);
                welcomeMessage.appendChild(timeDiv);
                
                chatMessages.appendChild(welcomeMessage);
            }
        } catch (error) {
            console.error('Error resetting conversation:', error);
            addMessage("Xin lỗi, không thể đặt lại cuộc trò chuyện. Vui lòng thử lại sau.");
        }
    }
    
    // Add event listeners if elements exist
    if (sendBtn) {
        sendBtn.addEventListener('click', () => {
            if (userInput) sendMessage(userInput.value);
        });
    }
    
    if (userInput) {
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage(userInput.value);
            }
        });
        
        // Auto-resize the textarea as the user types
        userInput.addEventListener('input', () => {
            userInput.style.height = 'auto';
            userInput.style.height = (userInput.scrollHeight < 120) ? userInput.scrollHeight + 'px' : '120px';
        });
    }
    
    if (resetBtn) {
        resetBtn.addEventListener('click', resetConversation);
    }

    // Floating chat button functionality
    const floatingBtn = document.getElementById('floatingChatBtn');
    const chatModal = document.getElementById('chatModal');
    const chatModalOverlay = document.getElementById('chatModalOverlay');
    
    if (floatingBtn && chatModal && chatModalOverlay) {
        floatingBtn.addEventListener('click', function() {
            chatModal.style.display = 'block';
            chatModalOverlay.style.display = 'block';
        });
        
        chatModalOverlay.addEventListener('click', function() {
            chatModal.style.display = 'none';
            chatModalOverlay.style.display = 'none';
        });
    }
});
