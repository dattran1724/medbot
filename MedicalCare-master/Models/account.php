<?php
require_once('./models/connection.php');
class Account
{
  var $conn;
  public function __construct()
  {
    $connection = new Connection();
    $this->conn = $connection->conn;
  }
  public function getAccounts()
  {
    $query = "SELECT * FROM user";
    $result = $this->conn->query($query)->fetchAll();
    $array = array();
    foreach ($result as $row) {
      $array[] = $row;
    }
    return $array;
  }
  public function checkAccount($username, $password)
  {
    // Prepare and execute a SQL query
    $query = "SELECT * FROM user WHERE username = :username AND password = :password";
    $stmt = $this->conn->prepare($query);

    // Bind parameters
    $stmt->bindParam(':username', $username);
    $stmt->bindParam(':password', $password);

    // Execute the statement
    $stmt->execute();

    // Fetch the result
    $user = $stmt->fetch();

    // Check if the user exists and the password is correct
    if ($user != null) {
      // Password is correct
      switch ($user['roleID']) {
        case 3:
          $_SESSION['isLogin']['admin'] = true;
          break;
        case 2;
          $_SESSION['isLogin']['doctor'] = true;
          break;
        case 1;
          $_SESSION['isLogin']['customer'] = true;
          break;
        default:
          break;
      }
      $_SESSION['user'] = $user;
      header('Location: ?act=home');
      exit(); // Important to stop script execution after redirect
    } else {
      $_SESSION['msg1'] = "Login not successful!";
      header('Location: ?act=account&process=login');
      exit(); // Important to stop script execution after redirect
    }
  }
  public function getAccountName($username, $password)
  {
    // Prepare and execute a SQL query
    $query = "SELECT * FROM user WHERE username = $username AND password = $password";
    $result = $this->conn->query($query)->fetch();
    return $result['name'];
  }
  function insertUser($name, $username, $password, $gender)
  {

    $query = "INSERT INTO user (name, username, password, gender, roleID) 
                  VALUES (:name, :username, :password, :gender,1)";
    $statement = $this->conn->prepare($query);
    $statement->bindParam(':name', $name);
    $statement->bindParam(':username', $username);
    $statement->bindParam(':password', $password);
    $statement->bindParam(':gender', $gender);
    $statement->execute();
  }

  function validateSignupForm($name, $username, $password, $rePassword, $gender, $email)
  {
    $errors = array();

    // Validate Name
    if (empty($name)) {
      $errors[] = "Chưa nhập họ tên";
    }

    // Validate Username
    if (empty($username)) {
      $errors[] = "Tên đăng nhập không được để trống.";
    } elseif (!preg_match('/^[a-zA-Z0-9_]+$/', $username)) {
      $errors[] = "Tên đăng nhập chỉ được chứa chữ cái, chữ số và dấu gạch dưới (_)";
    }

    // Validate Password
    if (empty($password)) {
      $errors[] = "Chưa nhập mật khẩu";
    } elseif (strlen($password) < 6) {
      $errors[] = "Mật khẩu phải dài hơn 6 kí tự";
    }

    // Validate Re-entered Password
    if ($password !== $rePassword) {
      $errors[] = "Mật khẩu nhập lại không khớp";
    }

    // Validate Gender
    if (!in_array($gender, array('Nam', 'Nữ'))) {
      $errors[] = "Giới tính không hợp lệ";
    }

    // Validate Email
    if (empty($email)) {
      $errors[] = "Chưa nhập địa chỉ email";
    } elseif (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
      $errors[] = "Địa chỉ email không hợp lệ";
    }

    if (empty($errors)) {
      $query = "INSERT INTO user (name, username, password, email, gender, roleID) 
            VALUES (:name, :username, :password, :email, :gender, 1)";
      $statement = $this->conn->prepare($query);
      $statement->bindParam(':name', $name);
      $statement->bindParam(':username', $username);
      $statement->bindParam(':password', $password);
      $statement->bindParam(':email', $email);
      $statement->bindParam(':gender', $gender);
      $statement->execute();
      $_SESSION['signup_check'] = true;
    } else {
      $_SESSION['errors'] = $errors;
    }

    header('Location: ?act=account&process=signup');
  }

  public function logout()
  {
    session_destroy();
    header('Location: ?act=account&process=login');
  }
  public function account()
  {
    require_once('./views/index.php');
  }
  public function admin($id)
  {
    $query = "SELECT * FROM user WHERE userID = :id AND roleID = 3";
    $stmt = $this->conn->prepare($query);
    $stmt->bindParam(':id', $id);
    $result = $stmt->execute();
    if ($result) {
      return true;
    } else {
      return false;
    }
  }
  public function updateUser($id, $name, $username, $password, $email, $gender)
  {
    $query = "UPDATE user
                SET name = :name,
                    username = :username,
                    password = :password,
                    email = :email,
                    gender = :gender
                WHERE userID = :id";

    $stmt = $this->conn->prepare($query);

    // Bind parameters
    $stmt->bindParam(':id', $id);
    $stmt->bindParam(':name', $name);
    $stmt->bindParam(':username', $username);
    $stmt->bindParam(':password', $password);
    $stmt->bindParam(':email', $email);
    $stmt->bindParam(':gender', $gender);

    // Execute the update
    $result = $stmt->execute();
    if ($result) {
      $_SESSION['msg'] = 'update_true';
    } else {
      $_SESSION['msg'] = 'update_false';
    }
    header('Location: ?act=account');
  }
  public function updatePass($email,  $password)
  {
    $query = "UPDATE user
                SET password = :password
                WHERE email = :email";

    $stmt = $this->conn->prepare($query);

    // Bind parameters
    $stmt->bindParam(':email', $email);
    $stmt->bindParam(':password', $password);



    // Execute the update
    $result = $stmt->execute();
    if ($result) {
      $_SESSION['msg'] = 'update_true';
    } else {
      $_SESSION['msg'] = 'update_false';
    }
    header('Location: ?act=account');
  }
}
