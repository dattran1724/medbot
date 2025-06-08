<?php
require_once('./models/account.php');
class AccountController
{
  var $account_model;
  public function __construct()
  {
    $this->account_model = new Account();
  }
  public function goback()
  {
    $this->account_model->goback();
  }
  public function list($id)
  {
    $accounts = $this->account_model->getAccounts($id);
    require_once('./views/index.php');
  }
  public function add()
  {
    $name = $_POST['name'];
    $username = $_POST['username'];
    $password = $_POST['password'];
    $email = $_POST['email'];
    $gender = $_POST['gender'];
    $role = $_POST['roleID'];
    $this->account_model->addAccount($name, $username, $password, $email, $gender, $role);
  }
  public function edit()
  {
    $id = filter_input(INPUT_GET, 'id', FILTER_VALIDATE_INT);
    $account = $this->account_model->getAccount($id);
    require_once('./views/index.php');
  }
  public function update()
  {
    $id = $_POST['id'];
    $name = $_POST['name'];
    $email = $_POST['email'];
    $username = $_POST['username'];
    $password = $_POST['password'];
    $gender = $_POST['gender'];
    $role = $_POST['roleID'];
    $this->account_model->update($id, $name, $username, $password, $email, $gender, $role);
  }
  public function delete()
  {
    $id = $_GET['id'];
    $this->account_model->delete($id);
  }
  public function logout() {
    $this->account_model->logout();
  }
}
