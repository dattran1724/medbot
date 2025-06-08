<?php
require_once('./models/database.php');
class Status
{
  var $db;
  public function __construct()
  {
    $database = new Database();
    $this->db = $database->db;
  }
  public function getStatus()
  {
    $query = "SELECT *FROM status ";
    require("result.php");
    return $data;
  }
  public function getStatusbyID($id)
  {
    $query = "SELECT * FROM status WHERE id_status = '$id'";

    require("result.php");
    return $data;
  }
  public function updateStatus($id, $ngay, $gio)
  {
    $query = "UPDATE status SET gio = '$gio', ngay = '$ngay' WHERE status.id_status = $id";

    $statement = $this->db->prepare($query);

    $success = $statement->execute();
    if ($success) {
        setcookie("msg", "Edit successful!", time() + 2);
      header('Location: ?act=' . 'status');
    } else {
        setcookie("msg", "Edit fail!", time() + 2);
      header('Location: ?act=' . 'status' . '&process=edit&id=' . $id);
    }

  }
  public function addStatus($ngay, $gio)
  {
    try {
      $query = "INSERT INTO status (ngay,gio) 
        VALUES (:ngay,:gio)";

      $statement = $this->db->prepare($query);
      $statement->bindParam(':ngay', $ngay);
      $statement->bindParam(':gio', $gio);

      $success = $statement->execute();
      echo $success;
      if ($success) {
        setcookie("msg", "Add successful!", time() + 2);
        header('Location: ?act=' . 'status');
      } 
    }catch (PDOException $e) {
        setcookie("msg", "Add fail!", time() + 2);
      header('Location: ?act=' . 'status&process=add');
      exit;
  }

  }
  public function deleteStatus($id)
  {
    try {
    $query = "DELETE FROM status WHERE `status`.`id_status` = $id";
    $statement = $this->db->prepare($query);
    $status= $statement->execute();
    if ($status == true) {
        header('Location: ?act=' . 'status');
        setcookie('msg', 'Delete successful!', time() + 2);
    }}catch (PDOException $e) {
        setcookie("msg", "Delete fail!", time() + 2);
      header('Location: ?act=' . 'status');
      exit;
  } 
  }
//   public function getstatus()
//   {
//     $query = "SELECT * FROM status  ";
//     require("result.php");
//     return $data;
//   }
}