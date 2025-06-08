<?php
require_once("Models/chuyengia.php");
class ChuyenGiaController
{
  var $chuyengia_model;

  public function __construct()
  {
    $this->chuyengia_model = new Chuyengia();
  }

  public function list()
  {
  //   $selectedLocation = isset($_POST['location']) ? $_POST['location'] : 'all';
  //   if ($selectedLocation !== 'all') {
  //     $data = $this->chuyengia_model->selectlocation($selectedLocation);
  // } else {
      
  // }
 
    $id = isset($_GET['trang']) ? $_GET['trang'] : 1;
    $limit = 6;
    $start = ($id - 1) * $limit;
    $data = $this->chuyengia_model->limit($start, $limit);

    $data_count = $this->chuyengia_model->list();
    $data_tong  = count($data_count);
    require_once('Views/index.php');
  }
}
