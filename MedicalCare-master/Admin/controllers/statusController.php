<?php
require_once('./models/status.php');
class StatusController
{
  var $status_model;
  public function __construct()
  {
    $this->status_model = new status();
  }
  public function list()
  {
    $data = $this->status_model->getStatus();
    require_once('./views/index.php');
  }
  public function edit()
  {
    if (isset($_GET['id'])) {
      $id = $_GET['id'];
      $data = $this->status_model->getStatusbyID($id);
    }
    require_once('./views/index.php');
  }
  public function update()
  {
    $id = $_GET['id'];
    $ngay = date_create($_POST['ngay']);
    $date = date_format($ngay,"d/m/Y");
    $giobatdau = $_POST['giobatdau'];
    $gioketthuc = $_POST['gioketthuc'];
    $gio =  $giobatdau . ' - ' . $gioketthuc ;

    $this->status_model->updateStatus($id,$date,$gio);
    
  }
  public function add()
  {

    require_once('./views/index.php');
  }
  public function store()
  {
    $ngay = $_POST['ngay'];
    $ngay = date_create($_POST['ngay']);
    $date = date_format($ngay,"d/m/Y");
    $giobatdau = $_POST['giobatdau'];
    $gioketthuc = $_POST['gioketthuc'];
    $gio =  $giobatdau . ' - ' . $gioketthuc ;
    $this->status_model->addStatus($date,$gio);
    // }else {
    //   setcookie("msg", "Add fail!", time() + 2);
    //   header('Location: ?act=' . 'status&process=add');
    // }
  }
  public function delete (){
    $id = $_GET['id'];
    $this->status_model->deletestatus($id);
  }

}