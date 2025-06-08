<?php
require_once('./models/connection.php');
class Cart
{
  var $conn;
  public function __construct()
  {
    $connection = new Connection();
    $this->conn = $connection->conn;
  }
  function getCart()
  {
    // Initialize the cart if it doesn't exist in the session
    if (!isset($_SESSION['cart'])) {
      $_SESSION['cart'] = array();
    }
    return $_SESSION['cart'];
  }
  public function addToCart($productId, $quantity = 1, $productName = '', $productImage = '', $productPrice = 0.0)
  {
    // Initialize the cart if it doesn't exist in the session
    if (!isset($_SESSION['cart'])) {
      $_SESSION['cart'] = array();
    }

    // Check if the product is already in the cart
    if (array_key_exists($productId, $_SESSION['cart'])) {
      // If the product is in the cart, increase the quantity
      $_SESSION['cart'][$productId]['quantity'] += $quantity;
    } else {
      // If the product is not in the cart, add it with the specified quantity, name, image, and price
      $_SESSION['cart'][$productId] = array(
        'quantity' => $quantity,
        'name' => $productName,
        'image' => $productImage,
        'price' => $productPrice
      );
    }
    header("Location: ?act=pharmacy");
  }
  public function deleteProductFromCart($productId)
  {
    // Check if the cart exists in the session
    if (isset($_SESSION['cart'])) {
      // Check if the product with the given ID exists in the cart
      if (array_key_exists($productId, $_SESSION['cart'])) {
        // Remove the product from the cart
        unset($_SESSION['cart'][$productId]);
      }
    }
    // Optionally, you can add a redirection or additional logic here
    header("Location: ?act=product-detail&process=list_order");
  }


  public function getCartInDrugID($cart)
  {
    $result = array();
    if (isset($cart) && !empty($cart)) {
      foreach ($cart as $productId => $quantity) {
        $result[] = $productId;
      }
    }
    return $result;
  }
  function updateQuantity($productId, $newQuantity)
  {
    // Check if the cart exists in the session
    if (isset($_SESSION['cart'])) {
      // Check if the product with the given ID exists in the cart
      if (array_key_exists($productId, $_SESSION['cart'])) {
        // Update the quantity for the specified product ID
        $_SESSION['cart'][$productId]['quantity'] = $newQuantity;
      }
    }
    // Optionally, you can add a redirection or additional logic here
    header("Location: ?act=product-detail&process=list_order");
  }
}
