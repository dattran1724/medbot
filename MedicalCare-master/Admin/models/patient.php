<?php
require_once('./models/database.php');
class Patient
{
  var $db;
  public function __construct()
  {
    $database = new Database();
    $this->db = $database->db;
  }
  public function getPatients()
  {
    $query = "SELECT * FROM benhnhan";
    require("result.php");
    return $data;
  }
  public function getPatientbyID($id)
  {
    $query = "SELECT * FROM benhnhan WHERE id_bn = '$id'";

    require("result.php");
    return $data;
  }
  public function updatePatient($id,$name,$sdt,$gioitinh,$ngaysinh,$tinh,$huyen,$xa,$diachi)
  {
    $query = "UPDATE benhnhan SET name = '$name',sdt = '$sdt', gioitinh = '$gioitinh', ngaysinh = '$ngaysinh', tinh = '$tinh',
     huyen = '$huyen', xa = '$xa ', diachi = '$diachi' WHERE benhnhan.id_bn = $id";

    $statement = $this->db->prepare($query);

    $success = $statement->execute();
    if ($success) {
      setcookie("msg","Edit successful!", time() + 2);
      header('Location: ?act=' . 'patient');
    } else {
      setcookie("msg", "Edit fail!", time() + 2);
      header('Location: ?act=' . 'patient' . '&process=edit&id=' . $id);
    }

  }
//   public function addPatient($userID,$name,$sdt,$gioitinh,$ngaysinh,$tinh,$huyen,$xa,$diachi)
//   {
//     try {
//         $query = "INSERT INTO benhnhan (userID,name,sdt,gioitinh, ngaysinh, tinh, huyen, xa, diachi) 
//         VALUES (:userID,:name,:sdt, :ngaysinh, :tinh, :huyen, :xa, :diachi)";
//         $statement = $this->conn->prepare($query);
//         $statement->bindParam(':userID', $userID);
//         $statement->bindParam(':name', $name);
//         $statement->bindParam(':sdt', $sdt);
//         $statement->bindParam(':gioitinh', $gioitinh);
//         $statement->bindParam(':ngaysinh', $ngaysinh);
//         $statement->bindParam(':tinh', $tinh);
//         $statement->bindParam(':huyen', $huyen);
//         $statement->bindParam(':xa', $xa);
//         $statement->bindParam(':diachi', $diachi);

//       $success = $statement->execute();
//       echo $success;
//       if ($success) {
//         setcookie("msg", "Add successful!", time() + 2);
//         header('Location: ?act=' . 'patient');
//       }
//     } catch (PDOException $e) {
//       setcookie("msg", "Add fail!", time() + 2);
//       header('Location: ?act=' . 'patient&process=add');
//       exit;
//     }

//   }
  public function deletePatient($id)
  { 
    try {
    $query = "DELETE FROM benhnhan WHERE `benhnhan`.`id_bn` = $id";
    $statement = $this->db->prepare($query);
    $status= $statement->execute();
    if ($status == true) {
      setcookie('msg', 'Delete successful!', time() + 2);
      header('Location: ?act=' . 'patient');
  }}catch (PDOException $e) {
      setcookie("msg", "Delete fail!", time() + 2);
    header('Location: ?act=' . 'patient');
    exit;
  }
}}