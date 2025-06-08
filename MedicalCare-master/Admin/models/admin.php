<?php
require_once('./models/database.php');
class Admin
{
  var $db;
  public function __construct()
  {
    $database = new Database();
    $this->db = $database->db;
  }
  public function delete($id)
  {
    // Use prepared statements to prevent SQL injection
    $query = "DELETE FROM user WHERE userID = :id";

    $statement = $this->db->prepare($query);
    $statement->bindParam(':userID', $id);

    // Execute the query
    $statement->execute();
  }

  public function update($id, $name, $username, $password, $address, $gender, $role)
  {
    // Use prepared statements to prevent SQL injection
    $query = "UPDATE user 
              SET name = :name, 
                  username = :username, 
                  password = :password, 
                  address = :address, 
                  gender = :gender, 
                  roleID = :role 
              WHERE userID = :id";

    $statement = $this->db->prepare($query);
    $statement->bindParam(':id', $id);
    $statement->bindParam(':name', $name);
    $statement->bindParam(':username', $username);
    $statement->bindParam(':password', $password);
    $statement->bindParam(':address', $address);
    $statement->bindParam(':gender', $gender);
    $statement->bindParam(':role', $role);

    // Execute the query
    $statement->execute();
  }

  public function add($name, $username, $password, $address, $gender, $role)
  {

    $query = "INSERT INTO user (name, username, password, address, gender, roleID) 
                  VALUES (:name, :username, :password, :address, :gender,:role)";
    $statement = $this->db->prepare($query);
    $statement->bindParam(':name', $name);
    $statement->bindParam(':username', $username);
    $statement->bindParam(':password', $password);
    $statement->bindParam(':address', $address);
    $statement->bindParam(':gender', $gender);
    $statement->bindParam(':role', $role);
    $statement->execute();
  }
}
