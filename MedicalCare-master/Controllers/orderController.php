<?php
require_once('./models/order.php');
class OrderController
{
  var $order_model;
  public function __construct()
  {
    $this->order_model = new Order();
  }

  public function listOrder()
  {
    require_once('./Views/index.php');
  }
  public function makeOrder()
  {
    if (isset($_SESSION['isLogin']) && $_SESSION['isLogin'] == true) {
      $user_id = (isset($_SESSION['user']['userID']) && $_SESSION['user']['userID'] != null) ? $_SESSION['user']['userID'] : null;
      $delivery_address = $_POST['home'] . "| " . $_POST['ward'] . "| " . $_POST['district'] . "| " . $_POST['city'];
      $orderID = $this->order_model->createOrder($user_id, $delivery_address);
      $_SESSION['msg'] = $this->order_model->makeOrder($orderID);
      require_once('./Views/index.php');
    }
  }
  public function viewOrder()
  {
    $user_id = (isset($_SESSION['user']['userID']) && $_SESSION['user']['userID'] != null) ? $_SESSION['user']['userID'] : null;
    $user_orders = $this->order_model->viewUserOrder($user_id);
    require_once('./Views/index.php');
  }
  public function viewOrderDetail()
  {
    $order_id = $_GET['id'];
    $orderDetails = $this->order_model->getOrderDetail($order_id);
    require_once('./Views/index.php');
  }
  public function deleteOrder() {
    $order_id = $_GET['id'];
    $this->order_model->deleteOrder($order_id);
  }
}
