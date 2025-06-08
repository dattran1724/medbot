<?php
require_once('./models/order.php');
class OrderController
{
  var $order_model;
  public function __construct()
  {
    $this->order_model = new Order();
  }
  public function list()
  {
    $orders = $this->order_model->getOrders();
    require_once('./views/index.php');
  }
  public function accept() {
    $order_id = $_GET['id'];
    $this->order_model->acceptOrder($order_id);
    header('Location: ?act=order');
  }
  public function orderDetail()
  {
    $order_id = $_GET['id'];
    $orderDetails = $this->order_model->getOrderDetail($order_id);
    require_once('./views/index.php');
  }
}
