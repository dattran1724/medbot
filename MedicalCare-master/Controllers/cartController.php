<?php
require_once('./models/cart.php');
class CartController
{
  var $cart_model;
  public function __construct()
  {
    $this->cart_model = new Cart();
  }
  public function list()
  {
    $this->cart_model->getCart();
  }
  public function add()
  {
    $id = $_POST['id'];
    $quantity = $_POST['quantity'];
    $name = $_POST['name'];
    $image = $_POST['image'];
    $price = $_POST['price'];
    $this->cart_model->addToCart($id, $quantity, $name, $image, $price);
    require_once('./Views/index.php');
  }
  public function delete()
  {
    $id = $_GET['id'];
    $this->cart_model->deleteProductFromCart($id);
  }
  public function update()
  {
    $id = $_POST['id'];
    $quantity = $_POST['quantity'];
    $this->cart_model->updateQuantity($id, $quantity);
  }
}
