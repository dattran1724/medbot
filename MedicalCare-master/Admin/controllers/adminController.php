<?php
require_once('./models/admin.php');
class AccountController
{
  var $admin_model;
  public function __construct()
  {
    $this->admin_model = new Admin();
  }

}