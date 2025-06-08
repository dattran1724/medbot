<?php
require_once('./models/database.php');
class Doctor
{
  var $db;
  public function __construct()
  {
    $database = new Database();
    $this->db = $database->db;
  }
  public function getDoctors()
  {
    $query = "SELECT *,b.name as name_bs, ck.name as chuyenkhoa  FROM bacsi as b , chuyenkhoa as ck WHERE b.id_ck = ck.id_ck";
    require("result.php");
    return $data;
  }
  public function getDoctorbyID($id)
  {
    $query = "SELECT * FROM bacsi WHERE id_bs = '$id'";

    require("result.php");
    return $data;
  }
  public function getDoctorbyUserID($id)
  {
    $query = "SELECT * FROM bacsi WHERE userID = '$id'";
    require("result.php");
    return $data;
  }
  public function updateDoctor($id, $name,$userID, $id_ck, $chucvu, $lamviec, $chuyenmon, $hocham, $hocvi, $thanhvien, $daotao, $kinhnghiem, $giakham, $hinhanh, $mota)
  {
    $query = "UPDATE bacsi SET userID = '$userID', chucvu = '$chucvu',id_ck = '$id_ck', hocham = '$hocham', hocvi = '$hocvi', chuyenmon = '$chuyenmon',
     mota = '$mota', thanhvien = '$thanhvien ', daotao = '$daotao', kinhnghiem = '$kinhnghiem',hinhanh = '$hinhanh',
      lamviec = '$lamviec', giakham = '$giakham', name = '$name' WHERE bacsi.id_bs = $id";

    $statement = $this->db->prepare($query);

    $success = $statement->execute();
    if ($success) {
      setcookie("msg","Edit successful!", time() + 2);
      header('Location: ?act=' . 'doctor');
    } else {
      setcookie("msg", "Edit fail!", time() + 2);
      header('Location: ?act=' . 'doctor' . '&process=edit&id=' . $id);
    }

  }
  public function addDoctor($name, $userID, $id_ck, $chucvu, $lamviec, $chuyenmon, $hocham, $hocvi, $thanhvien, $daotao, $kinhnghiem, $giakham, $hinhanh, $mota)
  {
    try {
      $query = "INSERT INTO bacsi (name,userID,id_ck,chucvu,lamviec,chuyenmon,hocham,hocvi,thanhvien,daotao,kinhnghiem,giakham,hinhanh,mota) 
        VALUES (:name,:userID,:id_ck,:chucvu,:lamviec,:chuyenmon,:hocham,:hocvi,:thanhvien,:daotao,:kinhnghiem,:giakham,:hinhanh,:mota)";

      $statement = $this->db->prepare($query);
      $statement->bindParam(':name', $name);
      $statement->bindParam(':userID', $userID);
      $statement->bindParam(':id_ck', $id_ck);
      $statement->bindParam(':chucvu', $chucvu);
      $statement->bindParam(':lamviec', $lamviec);
      $statement->bindParam(':chuyenmon', $chuyenmon);
      $statement->bindParam(':hocham', $hocham);
      $statement->bindParam(':hocvi', $hocvi);
      $statement->bindParam(':thanhvien', $thanhvien);
      $statement->bindParam(':daotao', $daotao);
      $statement->bindParam(':kinhnghiem', $kinhnghiem);
      $statement->bindParam(':giakham', $giakham);
      $statement->bindParam(':hinhanh', $hinhanh);
      $statement->bindParam(':mota', $mota);

      $success = $statement->execute();
      echo $success;
      if ($success) {
        setcookie("msg", "Add successful!", time() + 2);
        header('Location: ?act=' . 'doctor');
      }
    } catch (PDOException $e) {
      setcookie("msg", "Add fail!", time() + 2);
      header('Location: ?act=' . 'doctor&process=add');
      exit;
    }

  }
  public function getChuyenkhoa()
  {
    $query = "SELECT * FROM chuyenkhoa  ";
    require("result.php");
    return $data;
  }
  public function deleteDoctor($id)
  { 
    try {
    $query = "DELETE FROM bacsi WHERE `bacsi`.`id_bs` = $id";
    $statement = $this->db->prepare($query);
    $status= $statement->execute();
    if ($status == true) {
      setcookie('msg', 'Delete successful!', time() + 2);
      header('Location: ?act=' . 'doctor');
  }}catch (PDOException $e) {
      setcookie("msg", "Delete fail!", time() + 2);
    header('Location: ?act=' . 'doctor');
    exit;
  }
}}