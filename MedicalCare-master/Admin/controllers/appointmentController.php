<?php
require_once('./models/appointment.php');
class AppointmentController
{
  var $appointment_model;
  public function __construct()
  {
    $this->appointment_model = new appointment();
  }
  public function list()
  {
    
    if (isset($_GET['id_user'])) {
      $id = $_GET['id_user'];
      $data = $this->appointment_model->getAppointmentbyIDdoctor($id);
    }else {
      $data = $this->appointment_model->getAppointments();
    }
    require_once('./views/index.php');
  }
  public function detail()
  {
    if (isset($_GET['id'])) {
      $id = $_GET['id'];
      $data = $this->appointment_model->getAppointmentbyID($id);
    }
    require_once('./views/index.php');
  }
  // public function update()
  // {
  //   $id = $_GET['id'];
  //   $id_bn = $_POST['id_bn'];
  //   $id_bs = $_POST['id_bs'];
  //   $id_kh = $_POST['id_kh'];
  //   $lydo = $_POST['lydo'];
  //   $trangthai = $_POST['trangthai'];
  //   $this->appointment_model->updateAppointment($id,$id_bn,$id_bs,$id_kh,$lydo,$trangthai);
    
  // }
  public function delete (){
    $id = $_GET['id'];
    $this->appointment_model->deleteAppointment($id);
  }

  public function accept (){
    $id = $_GET['id'];
    $this->appointment_model->acceptAppointment($id);
  }


}