<?php
require_once('./models/database.php');
class Specialist
{
  var $db;
  public function __construct()
  {
    $database = new Database();
    $this->db = $database->db;
  }
  public function getSpecialists()
  {
    $query = "SELECT * FROM chuyenkhoa";
    require("result.php");
    return $data;
  }
  public function getSpecialistbyID($id)
  {
    $query = "SELECT * FROM chuyenkhoa WHERE id_ck = '$id'";

    require("result.php");
    return $data;
  }
  // public function getSpecialistbyUserID($id)
  // {
  //   $query = "SELECT * FROM bacsi WHERE userID = '$id'";
  //   require("result.php");
  //   return $data;
  // }
  public function updatespecialist($id, $name, $hinhanh, $mota)
  {
    $query = "UPDATE chuyenkhoa SET mota = '$mota', hinhanh = '$hinhanh ',name = '$name' WHERE chuyenkhoa.id_ck = $id";

    $statement = $this->db->prepare($query);

    $success = $statement->execute();
    if ($success) {
      setcookie("msg", "Edit successful!", time() + 2);
      header('Location: ?act=' . 'specialist');
    } else {
      setcookie("msg", "Edit fail!", time() + 2);
      header('Location: ?act=' . 'specialist' . '&process=edit&id=' . $id);
    }

  }
  public function addspecialist($name, $hinhanh, $mota)
  {
    try {
      $query = "INSERT INTO chuyenkhoa (name,hinhanh,mota) 
        VALUES (:name,:hinhanh,:mota)";

      $statement = $this->db->prepare($query);
      $statement->bindParam(':name', $name);
      $statement->bindParam(':hinhanh', $hinhanh);
      $statement->bindParam(':mota', $mota);

      $success = $statement->execute();
      if ($success) {
        setcookie("msg", "Add successful!", time() + 2);
        header('Location: ?act=' . 'specialist');
      }
    } catch (PDOException $e) {
      setcookie("msg", "Add fail!", time() + 2);
      header('Location: ?act=' . 'specialist&process=add');
      exit;
    }

  }
  public function deletespecialist($id)
  { 
    try {
    $query = "DELETE FROM chuyenkhoa WHERE `chuyenkhoa`.`id_ck` = $id";
    $statement = $this->db->prepare($query);
    $status= $statement->execute();
    if ($status == true) {
      setcookie('msg', 'Delete successful!', time() + 2);
      header('Location: ?act=' . 'specialist');
  }}catch (PDOException $e) {
      setcookie("msg", "Delete fail!", time() + 2);
    header('Location: ?act=' . 'specialist');
    exit;
  }
}}