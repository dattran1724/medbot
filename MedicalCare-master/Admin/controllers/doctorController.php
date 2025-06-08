<?php
require_once('./models/doctor.php');
class DoctorController
{
  var $doctor_model;
  public function __construct()
  {
    $this->doctor_model = new Doctor();
  }
  public function list()
  {
    $data = $this->doctor_model->getDoctors();
    
    require_once('./views/index.php');
  }
  public function edit()
  {
    $data_ck = $this->doctor_model->getChuyenkhoa();
    if (isset($_GET['id'])) {
      $id = $_GET['id'];
      $data = $this->doctor_model->getDoctorbyID($id);
    }
    require_once('./views/index.php');
  }
  public function update()
  {
    $id = $_GET['id'];
    $name = $_POST['name'];
    $userID = $_POST['userID'];
    $id_ck = $_POST['id_ck'];
    $chucvu = $_POST['chucvu'];
    $lamviec = $_POST['lamviec'];
    $chuyenmon = $_POST['chuyenmon'];
    $hocham = $_POST['hocham'];
    $hocvi = $_POST['hocvi'];
    $thanhvien = $_POST['thanhvien'];
    $daotao = $_POST['daotao'];
    $kinhnghiem = $_POST['kinhnghiem'];
    $giakham = $_POST['giakham'];
    $hinhanh = $_POST['hinhanh'];
    $mota = $_POST['mota'];

    $this->doctor_model->updateDoctor($id,$name,$userID,$id_ck,$chucvu,$lamviec,$chuyenmon,$hocham,$hocvi,$thanhvien,$daotao,$kinhnghiem,$giakham,$hinhanh,$mota);
    
  }
  public function add()
  {
    $data_ck = $this->doctor_model->getChuyenkhoa();

    require_once('./views/index.php');
  }
  public function store()
  {
    $name = $_POST['name'];
    $userID = $_POST['userID'];
    $id_ck = $_POST['id_ck'];
    $chucvu = $_POST['chucvu'];
    $lamviec = $_POST['lamviec'];
    $chuyenmon = $_POST['chuyenmon'];
    $hocham = $_POST['hocham'];
    $hocvi = $_POST['hocvi'];
    $thanhvien = $_POST['thanhvien'];
    $daotao = $_POST['daotao'];
    $kinhnghiem = $_POST['kinhnghiem'];
    $giakham = $_POST['giakham'];
    $hinhanh = $_POST['hinhanh'];
    $mota = $_POST['mota'];
    $doctor = $this->doctor_model->getDoctorbyUserID($userID);
    if (empty($doctor)&& $userID !=NULL) {
    $this->doctor_model->addDoctor($name,$userID,$id_ck,$chucvu,$lamviec,$chuyenmon,$hocham,$hocvi,$thanhvien,$daotao,$kinhnghiem,$giakham,$hinhanh,$mota);
    }else {
      setcookie("msg", "Add fail!", time() + 2);
      header('Location: ?act=' . 'doctor&process=add');
    }
  }
  public function delete (){
    $id = $_GET['id'];
    $this->doctor_model->deleteDoctor($id);
  }

}