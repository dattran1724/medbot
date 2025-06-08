<?php
require_once('./models/specialist.php');
class SpecialistController
{
  var $specialist_model;
  public function __construct()
  {
    $this->specialist_model = new Specialist();
  }
  public function list()
  {
    $data = $this->specialist_model->getspecialists();
    require_once('./views/index.php');
  }
  public function edit()
  {

    if (isset($_GET['id'])) {
      $id = $_GET['id'];
      $data = $this->specialist_model->getspecialistbyID($id);
    }
    require_once('./views/index.php');
  }
  public function update()
  {
    $id = $_GET['id'];
    $name = $_POST['name'];
    $hinhanh = $_POST['hinhanh'];
    $mota = $_POST['mota'];

    $this->specialist_model->updatespecialist($id,$name,$hinhanh,$mota);
    
  }
  public function add()
  {
    

    require_once('./views/index.php');
  }
  public function store()
  {
    $name = $_POST['name'];
    $hinhanh = $_POST['hinhanh'];
    $mota = $_POST['mota'];
    // $specialist = $this->specialist_model->getspecialistbyUserID($userID);
    // if (empty($specialist)&& $userID !=NULL) {
    $this->specialist_model->addspecialist($name,$hinhanh,$mota);
    // }else {
    //   setcookie("msg", "Add fail!", time() + 2);
    //   header('Location: ?act=' . 'specialist&process=add');
    // }
  }
  public function delete (){
    $id = $_GET['id'];
    $this->specialist_model->deletespecialist($id);
  }
}