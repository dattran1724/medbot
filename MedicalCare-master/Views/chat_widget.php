<!-- Floating chat button -->
<div class="floating-chat-btn" id="floatingChatBtn">
  <i class="fas fa-comment-medical"></i>
</div>

<!-- Chat modal -->
<div class="chat-modal medical-chat-widget" id="chatModal">
  <div class="chat-container">
    <div class="chat-header">
      <h2><i class="fas fa-user-md"></i> Trợ lý Y tế</h2>
      <button id="resetBtn" title="Bắt đầu lại cuộc trò chuyện"><i class="fas fa-redo"></i></button>
    </div>
    <div class="chat-messages" id="chatMessages">
      <div class="message assistant">
        <div class="message-content">
          Xin chào! Tôi là trợ lý y tế MedicalCare. Tôi có thể giúp gì cho bạn hôm nay?
        </div>
        <div class="message-time">Vừa xong</div>
      </div>
    </div>
    <div class="chat-input">
      <textarea id="userInput" placeholder="Nhập câu hỏi của bạn tại đây..."></textarea>
      <button id="sendBtn"><i class="fas fa-paper-plane"></i></button>
    </div>
  </div>
</div>

<!-- Chat modal overlay -->
<div class="chat-modal-overlay" id="chatModalOverlay"></div>
