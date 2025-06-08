<?php
require_once('./models/database.php');
class Account
{
  var $db;
  public function __construct()
  {
    $database = new Database();
    $this->db = $database->db;
  }
  public function goback()
  {
    header("Location: ../index.php?act=pharmacy");
  }
  public function logout()
  {
    session_destroy();
    header('Location: ../index.php?act=account&process=login');
  }
  public function getAccounts($role_id)
  {
    // Use prepared statements to prevent SQL injection
    $query = "SELECT * FROM user WHERE roleID = $role_id";

    $statement = $this->db->prepare($query);
    $statement->execute();

    // Fetch all results
    $result = $statement->fetchAll(PDO::FETCH_ASSOC);

    // Return the array of user accounts
    return $result;
  }
  public function addAccount($name, $username, $password, $email, $gender, $role)
  {
    $query = "INSERT INTO user (name, username, password, email, gender, roleID) 
                  VALUES (:name, :username, :password, :email, :gender,:role)";
    $statement = $this->db->prepare($query);
    $statement->bindParam(':name', $name);
    $statement->bindParam(':username', $username);
    $statement->bindParam(':password', $password);
    $statement->bindParam(':email', $email);
    $statement->bindParam(':gender', $gender);
    $statement->bindParam(':role', $role);
    $result = $statement->execute();
    if ($result) {
      $_SESSION['msg'] = 'add_true';
    } else {
      $_SESSION['msg'] = 'add_false';
    }
    switch ($role) {
      case 1:
        header('Location: ?act=account&role=customer');
        break;
      case 2:
        header('Location: ?act=account&role=doctor');
        break;
      case 3:
        header('Location: ?act=account&role=admin');
        break;
      default:
        break;
    }
  }
  public function getAccount($id)
  {
    $query = "SELECT * FROM user WHERE userID = $id";
    $statement = $this->db->prepare($query);
    $statement->execute();

    // Fetch all results
    $result = $statement->fetch();

    // Return the array of user accounts
    return $result;
  }
  public function update($id, $name, $username, $password, $email, $gender, $role)
  {
    $query = "UPDATE user
              SET name = :name,
                  username = :username,
                  password = :password,
                  email = :email,
                  gender = :gender,
                  roleID = :role
              WHERE userID = :id";

    $stmt = $this->db->prepare($query);

    // Bind parameters
    $stmt->bindParam(':id', $id);
    $stmt->bindParam(':name', $name);
    $stmt->bindParam(':username', $username);
    $stmt->bindParam(':password', $password);
    $stmt->bindParam(':email', $email);
    $stmt->bindParam(':gender', $gender);
    $stmt->bindParam(':role', $role);

    // Execute the update
    $result = $stmt->execute();
    if ($result) {
      $_SESSION['msg'] = 'update_true';
    } else {
      $_SESSION['msg'] = 'update_false';
    }
    header('Location: ?act=account&role=customer');
  }
  public function delete($id)
  {
    $query = "DELETE FROM user WHERE userID = :id";

    $stmt = $this->db->prepare($query);

    // Bind parameter
    $stmt->bindParam(':id', $id);

    // Execute the delete
    $result = $stmt->execute();

    if ($result) {
      $_SESSION['msg'] = 'del_true';
    } else {
      $_SESSION['msg'] = 'del_false';
    }
    // Redirect or perform any other action after deletion
    header('Location: ?act=account&role=customer');
  }
}
