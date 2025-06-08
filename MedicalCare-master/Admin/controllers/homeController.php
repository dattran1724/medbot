<?php
require_once('./models/home.php');
class HomeController
{
  var $home_model;
  public function __construct()
  {
    $this->home_model = new Home();
  }
  public function list() {
    $total_account = $this->home_model->totalAccounts();
    $total_drug = $this->home_model->totalDrugs();
    $total_request = $this->home_model->totalRequests();
    $total_pending = $this->home_model->totalPending();
    require_once('./views/index.php');
  }
}
