<?php
require_once('./models/plan.php');
class PlanController
{
  var $plan_model;
  public function __construct()
  {
    $this->plan_model = new Plan();
  }
  public function list()
  {
    $data_status= $this->plan_model->getStatus();
    $data_bs= $this->plan_model->getDoctor();
    $data = $this->plan_model->getPlan();
    require_once('./views/index.php');
  }
  public function edit()
  {
    $data_status= $this->plan_model->getStatus();
    $data_bs= $this->plan_model->getDoctor();
    if (isset($_GET['id'])) {
      $id = $_GET['id'];
      $data = $this->plan_model->getPlanbyID($id);
    }
    require_once('./views/index.php');
  }
  public function update()
  {
    $id = $_GET['id'];
    // $id_bs = $_POST['id_bs'];
    $id_status = $_POST['id_status'];

    $this->plan_model->updatePlan($id,$id_status);
    
  }
  public function add()
  {
    $data_status= $this->plan_model->getStatus();
    $data_bs= $this->plan_model->getDoctor();
    require_once('./views/index.php');
  }
  public function store()
  {
    $id_bs = $_POST['id_bs'];
    $id_status = $_POST['id_status'];
    $this->plan_model->addPlan($id_bs, $id_status);
  }
  public function delete (){
    $id = $_GET['id'];
    $this->plan_model->deleteplan($id);
  }

}