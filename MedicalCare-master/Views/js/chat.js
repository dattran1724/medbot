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
        if (!text.trim() || isWaiting) return;
        
        isWaiting = true;
        sendBtn.disabled = true;
        
        // Add user message to chat
        addMessage(text, true);
        userInput.value = '';
        
        showTypingIndicator();
        
        try {
            const response = await fetch('/chat', {
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
                addMessage("Sorry, there was an error: " + data.error);
            }
        } catch (error) {
            console.error('Error:', error);
            hideTypingIndicator();
            addMessage("Sorry, I couldn't connect to the server. Please try again later.");
        }
        
        isWaiting = false;
        sendBtn.disabled = false;
    }
    
    // Reset the conversation
    async function resetConversation() {
        try {
            const response = await fetch('/reset', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Clear chat messages except the first welcome message
                while (chatMessages.childNodes.length > 1) {
                    chatMessages.removeChild(chatMessages.lastChild);
                }
                
                // Clear thread ID
                threadId = null;
                localStorage.removeItem('medicalCareThreadId');
                
                addMessage("Conversation has been reset. How can I help you today?");
            }
        } catch (error) {
            console.error('Error resetting conversation:', error);
            addMessage("Sorry, I couldn't reset the conversation. Please try again later.");
        }
    }
    
    // Event listeners
    sendBtn.addEventListener('click', () => {
        sendMessage(userInput.value);
    });
    
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage(userInput.value);
        }
    });
    
    resetBtn.addEventListener('click', resetConversation);
    
    // Auto-resize the textarea as the user types
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = (userInput.scrollHeight < 120) ? userInput.scrollHeight + 'px' : '120px';
    });
});
