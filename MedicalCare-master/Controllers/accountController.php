<?php
require_once('./models/account.php');
class AccountController
{
  var $account_model;
  public function __construct()
  {
    $this->account_model = new Account();
  }
  public function checkAccount()
  {
    $username = $_POST['username'];
    $password = $_POST['password'];
    $check = $this->account_model->checkAccount($username, $password);
  }
  public function list()
  {
    require_once('./views/index.php');
  }
  public function logout()
  {
    $this->account_model->logout();
  }
  public function account()
  {
    require_once('./views/index.php');
  }
  public function admin()
  {
    $ad_name = $_POST['admin_name'];
    $id = $_POST['user_id'];
    $check = $this->account_model->admin($id);
    if ($check) {
      header("Location: ./admin/index.php");
    } else {
      header("Location: ?act=error");
    }
  }
  public function updateUser()
  {
    $id = $_POST['id'];
    $name = $_POST['name'];
    $username = $_POST['username'];
    $password = $_POST['password'];
    $email = $_POST['email'];
    $gender = $_POST['gender'];
    $this->account_model->updateUser($id, $name, $username, $password, $email, $gender);
  }
}
