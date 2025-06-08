<?php
require_once('./models/account.php');
require("./phpmailer/src/PHPMailer.php");
require("./phpmailer/src/SMTP.php");
require("./phpmailer/src/Exception.php");
class LoginController
{
  var $account_model;
  public function __construct()
  {
    $this->account_model = new Account();
  }
  public function list()
  {
    require_once('Views/index.php');
  }
  public function logout()
  {
    $this->account_model->logout();
  }
  public function login()
  {
    require_once('Views/index.php');
  }
  public function forgot()
  {
    require_once('Views/index.php');
  }
  public function rand_string($length = 8) {
    $characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    $randomString = '';

    try {
        $bytes = random_bytes($length);
        for ($i = 0; $i < $length; $i++) {
            $randomString .= $characters[ord($bytes[$i]) % strlen($characters)];
        }
    } catch (Exception $e) {
        // Handle exception if random_bytes() fails
        // You might want to provide a fallback mechanism or log the error
        echo "Error generating random string: " . $e->getMessage();
    }

    return $randomString;
}
  public function sendEmail()
  {
    $randomPassword = $this->rand_string();
    $this->account_model->updatePass($_POST['email'],$randomPassword);
    $mail = new PHPMailer\PHPMailer\PHPMailer();
    $mail->IsSMTP(); // enable SMTP
    
    $mail->SMTPDebug = 1; // debugging: 1 = errors and messages, 2 = messages only
    $mail->SMTPAuth = true; // authentication enabled
    $mail->SMTPSecure = 'ssl'; // secure transfer enabled REQUIRED for Gmail
    $mail->Host = "smtp.gmail.com";
    $mail->Port = 465; // or 587
    $mail->IsHTML(true);
    $mail->Username = "huylvb.22ds@vku.udn.vn";
    $mail->Password = "gvshsbyfarizhvzs";
    $mail->SetFrom('huylvb.22ds@vku.udn.vn','Medical Care');
    $mail->Subject = "Reset Password";
    $content = "<h3>Dear</h3> ";
    $content .= "<p>We have received a request to re-issue your password recently</p>" ;
    $content .= "<p>We have updated and sent you a new password</p>" ;
    $content .= "<p>Your new password is: </p> <b> $randomPassword </b>" ;
    $mail->Body = $content;
    $mail->AddAddress($_POST['email']);
   header("Location: ?act=account&process=login");     
    if(!$mail->Send()) {
      echo "Mailer Error: " . $mail->ErrorInfo;
    
   } else {
      echo "Message has been sent";
   }

  }
  public function login_action()
  {
    $username = $_POST['username'];
    $password = $_POST['password'];
    // Check and redirect
    $this->account_model->checkAccount($username, $password);
  }
  public function signup()
  {
    require_once('Views/index.php');
  }
  public function signup_action()
  {
    $name = $_POST['name'];
    $username =  $_POST['username'];
    $password = $_POST['password'];
    $repassword = $_POST['repassword'];
    $email = $_POST['email'];
    $gender =  $_POST['gender'];
    $this->account_model->validateSignupForm($name, $username, $password, $repassword, $gender, $email);
  }
  
}
