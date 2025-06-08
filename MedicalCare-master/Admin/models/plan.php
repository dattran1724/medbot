<?php
require_once('./models/database.php');
class plan
{
  var $db;
  public function __construct()
  {
    $database = new Database();
    $this->db = $database->db;
  }
  public function getPlan()
  {
    $query = "SELECT k.* , b.name , s.* FROM kehoach as k , status as s , bacsi as b where k.id_status = s.id_status and b.id_bs = k.id_bs  ";
    require("result.php");
    return $data;
  }
  public function getPlanbyID($id)
  {
    $query = "SELECT k.* , b.name , s.* FROM kehoach as k , status as s , bacsi as b where k.id_status = s.id_status and b.id_bs = k.id_bs AND k.id_kh = $id ";

    require("result.php");
    return $data;
  }
  public function getStatus(){
    $query = "SELECT * FROM status " ;
    require("result.php");
    return $data;
  }
  public function getDoctor(){
    $query = "SELECT * FROM bacsi " ;
    require("result.php");
    return $data;
  }
  public function updatePlan($id_kh, $id_status)
  {
    $query = "UPDATE kehoach SET id_status = '$id_status' WHERE kehoach.id_kh = $id_kh";

    $statement = $this->db->prepare($query);

    $success = $statement->execute();
    if ($success) {
        setcookie("msg", "Edit successful!", time() + 2);
      header('Location: ?act=' . 'plan');
    } else {
        setcookie("msg", "Edit fail!", time() + 2);
      header('Location: ?act=' . 'plan' . '&process=edit&id=' . $id_kh);
    }

  }
  public function addPlan($id_bs, $id_status)
  {
    try {
      $query = "INSERT INTO kehoach (id_bs,id_status,trangthai) 
        VALUES (:id_bs,:id_status,:trangthai)";
        $trangthai = 0 ;
      $statement = $this->db->prepare($query);
      $statement->bindParam(':id_bs', $id_bs);
      $statement->bindParam(':id_status', $id_status);
      $statement->bindParam(':trangthai', $trangthai);

      $success = $statement->execute();
      echo $success;
      if ($success) {
        setcookie("msg", "Add successful!", time() + 2);
        header('Location: ?act=' . 'plan');
      } 
    }catch (PDOException $e) {
        setcookie("msg", "Add fail!", time() + 2);
      header('Location: ?act=' . 'plan&process=add');
      exit;
  }

  }
  public function deletePlan($id)
  {
    try {
    $query = "DELETE FROM kehoach WHERE `kehoach`.`id_kh` = $id";
    $statement = $this->db->prepare($query);
    $status= $statement->execute();
    if ($status == true) {
        header('Location: ?act=' . 'plan');
        setcookie('msg', 'Delete successful!', time() + 2);
    }}catch (PDOException $e) {
        setcookie("msg", "Delete fail!", time() + 2);
      header('Location: ?act=' . 'plan');
      exit;
  } 
  }
}