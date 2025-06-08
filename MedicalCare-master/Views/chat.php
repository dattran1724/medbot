<!DOCTYPE html>
<html lang="vi-vn">
<head>
  <meta charset="utf-8">
  <meta http-equiv="x-ua-compatible" content="ie=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Trợ lý Y tế - Medical Care</title>
  <link rel="icon" type="image/png" href="./public/img/images/logo.png" />
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" />
  <link rel="stylesheet" href="./public/css/navbar.css" />
  <link rel="stylesheet" href="./public/css/footer.css" />
  <link rel="stylesheet" href="./public/css/chat.css" />
</head>
<body class="chat-page">
  <!-- header section start -->
  <?php require_once("header_footer/header.php") ?>
  <!-- header section end -->

  <div class="main-chat-content medical-chat-widget">
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

  <!-- footer section start -->
  <?php require_once("header_footer/footer.php") ?>
  <!-- footer section end -->

  <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.0/jquery.min.js"></script>
  <script src="./public/js/chat.js"></script>
</body>
</html>
