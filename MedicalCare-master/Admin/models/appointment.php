<?php
require_once('./models/database.php');
class Appointment
{
  var $db;
  public function __construct()
  {
    $database = new Database();
    $this->db = $database->db;
  }
  public function getAppointments()
  {
    $query = "SELECT * , bs.name as name_bs , bn.name as name_bn, l.trangthai as trangthai_lh FROM lichhen as l, kehoach as k , status as s , bacsi as bs , benhnhan as bn 
    WHERE k.id_kh = l.id_kh AND k.id_status = s.id_status AND l.id_bs = bs.id_bs AND l.id_bn = bn.id_bn";
    require("result.php");
    return $data;
  }
  public function getDetail($id_kh,$id_bn)
  {
    $query = "SELECT * FROM kehoach as k , status as s , bacsi as bs , benhnhan as bn 
    WHERE k.id_kh = $id_kh AND k.id_status = s.id_status AND k.id_bs = bs.id_bs AND $id_bn = bn.id_bn ";
    require("result.php");
    return $data;
  }
  public function getAppointmentbyID($id)
  {
    $query = "SELECT * , bs.name as name_bs , bn.name as name_bn, l.trangthai as trangthai_lh FROM lichhen as l, kehoach as k , status as s , bacsi as bs , benhnhan as bn 
    WHERE k.id_kh = l.id_kh AND k.id_status = s.id_status AND l.id_bs = bs.id_bs AND l.id_bn = bn.id_bn AND l.id_lh = '$id'";

    require("result.php");
    return $data;
  }
  public function getAppointmentbyIDdoctor($id)
  {
    $query = "SELECT * , bs.name as name_bs , bn.name as name_bn, l.trangthai as trangthai_lh FROM lichhen as l, kehoach as k , status as s , bacsi as bs , benhnhan as bn ,user as u
    WHERE k.id_kh = l.id_kh AND k.id_status = s.id_status AND l.id_bs = bs.id_bs AND l.id_bn = bn.id_bn AND u.userID = bs.userID AND u.userID = '$id'";
    require("result.php");
    return $data;
  }
  public function updateAppointment($id, $id_bn, $id_bs, $id_kh, $lydo, $trangthai)
  {
    $query = "UPDATE lichhen SET id_bn = '$id_bn',id_bs = '$id_bs', id_kh = '$id_kh', lydo = '$lydo', trangthai = '$trangthai', WHERE id.lh = $id";

    $statement = $this->db->prepare($query);

    $success = $statement->execute();
    if ($success) {
      header('Location: ?act=' . 'appointment');
    } else {
      header('Location: ?act=' . 'appointment' . '&process=edit&id=' . $id);
    }

  }
  public function addAppointment($id_bn, $id_bs, $id_kh, $lydo, $trangthai)
  {
    try {
      $query = "INSERT INTO lichhen (id_bn,id_bs,id_kh,lydo,trangthai) 
        VALUES ($id_bn,$id_bs,$id_kh,$lydo,$trangthai)";

      $statement = $this->db->prepare($query);
      $statement->bindParam(':id_bn', $id_bn);
      $statement->bindParam(':id_bs', $id_bs);
      $statement->bindParam(':id_kh', $id_kh);
      $statement->bindParam(':lydo', $lydo);
      $statement->bindParam(':trangthai', $trangthai);
      $success = $statement->execute();
      echo $success;
      if ($success) {
        header('Location: ?act=' . 'appointment');
      }
    } catch (PDOException $e) {
      header('Location: ?act=' . 'appointment');
      exit;
    }

  }
  public function deleteAppointment($id)
  { 
    try {
    $query = "DELETE FROM lichhen WHERE `lichhen`.`id_lh` = $id";
    $statement = $this->db->prepare($query);
    $status= $statement->execute();
    if ($status == true) {
      setcookie('msg', 'Delete successful!', time() + 2);
      header('Location: ?act=' . 'appointment'); 
  }}catch (PDOException $e) {
      setcookie("msg", "Delete fail!", time() + 2);
    header('Location: ?act=' . 'appointment');
    exit;
  }}
  public function acceptAppointment($id)
  { 
    try {
    $query = "UPDATE lichhen as lh , kehoach as k  SET lh.trangthai = 1 , k.trangthai = 1 WHERE lh.`id_lh` = $id and lh.id_kh = k.id_kh ";
    $statement = $this->db->prepare($query);
    $status= $statement->execute();
    if ($status == true) {
      setcookie('msg', 'Accept successful!', time() + 2);
      header('Location: ?act=' . 'appointment'); 
  }}catch (PDOException $e) {
      setcookie("msg", "Accept fail!", time() + 2);
    header('Location: ?act=' . 'appointment');
    exit;
  }}
}