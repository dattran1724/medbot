<?php
require_once('./models/patient.php');
class PatientController
{
  var $patient_model;
  public function __construct()
  {
    $this->patient_model = new Patient();
  }
  public function list()
  {
    $data = $this->patient_model->getpatients();
    require_once('./views/index.php');
  }
  public function edit()
  {
 
    if (isset($_GET['id'])) {
      $id = $_GET['id'];
      $data = $this->patient_model->getPatientbyID($id);
    }
    require_once('./views/index.php');
  }
  public function update()
  {
    $id = $_GET['id'];
    $name = $_POST['name'];
    // $userID = $_POST['userID'];
    $sdt = $_POST['sdt'];
    $gioitinh = $_POST['gioitinh'];
    $ngaysinh = $_POST['ngaysinh'];
    $tinh = $_POST['tinh'];
    $huyen = $_POST['huyen'];
    $xa = $_POST['xa'];
    $diachi = $_POST['diachi'];

    $this->patient_model->updatePatient($id,$name,$sdt,$gioitinh,$ngaysinh,$tinh,$huyen,$xa,$diachi);
    
  }
  public function add()
  {
    // $data_ck = $this->patient_model->getChuyenkhoa();

    require_once('./views/index.php');
  }
//   public function store()
//   {
//     $name = $_POST['name'];
//     $userID = $_POST['userID'];
//     $id_ck = $_POST['id_ck'];
//     $chucvu = $_POST['chucvu'];
//     $lamviec = $_POST['lamviec'];
//     $chuyenmon = $_POST['chuyenmon'];
//     $hocham = $_POST['hocham'];
//     $hocvi = $_POST['hocvi'];
//     $thanhvien = $_POST['thanhvien'];
//     $daotao = $_POST['daotao'];
//     $kinhnghiem = $_POST['kinhnghiem'];
//     $giakham = $_POST['giakham'];
//     $hinhanh = $_POST['hinhanh'];
//     $mota = $_POST['mota'];
//     $patient = $this->patient_model->getpatientbyUserID($userID);
//     if (empty($patient)&& $userID !=NULL) {
//     $this->patient_model->addpatient($name,$userID,$id_ck,$chucvu,$lamviec,$chuyenmon,$hocham,$hocvi,$thanhvien,$daotao,$kinhnghiem,$giakham,$hinhanh,$mota);
//     }else {
//       setcookie("msg", "Add fail!", time() + 2);
//       header('Location: ?act=' . 'patient&process=add');
//     }
//   }
  public function delete (){
    $id = $_GET['id'];
    $this->patient_model->deletePatient($id);
  }

}