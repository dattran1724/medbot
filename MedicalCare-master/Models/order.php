<?php
require_once('./models/connection.php');
class Order
{
  var $conn;
  public function __construct()
  {
    $connection = new Connection();
    $this->conn = $connection->conn;
  }
  public function createOrder($user_id, $delivery_address)
  {
    if ($user_id == null) {
      return false;
    } else {
      $query = "INSERT INTO `order` (UserID, OrderDate,DeliveryAddress) VALUES (:user_id, NOW(),:address)";
      $stmt = $this->conn->prepare($query);
      $stmt->bindParam(':user_id', $user_id);
      $stmt->bindParam(':address', $delivery_address);

      $result = $stmt->execute();
      if ($result == true) {
        $orderID = (int) $this->conn->lastInsertId();
        return $orderID;
      } else {
        return false;
      }
    }
  }

  public function makeOrder($orderID)
  {
    if (!empty($_SESSION['cart'])) {
      // Iterate through each item in the cart and insert into order_detail
      foreach ($_SESSION['cart'] as $drugID => $productInfo) {
        $quantity = $productInfo['quantity'];
        $price = $productInfo['price']; // Assuming you have 'price' in the cart

        // Insert into order_detail table
        $sql = "INSERT INTO order_details (OrderID, DrugID, Quantity, Price) VALUES (?, ?, ?, ?)";
        $stmt = $this->conn->prepare($sql);
        $stmt->execute([$orderID, $drugID, $quantity, $price]);
      }
      return true;
    } else {
      return false;
    }
  }
  public function viewUserOrder($userID)
  {
    $query = "SELECT o.*, SUM(od.Price) AS TotalPrice
    FROM `order` o
    LEFT JOIN order_details od ON o.OrderID = od.OrderID
    WHERE o.UserID = :userID
    GROUP BY o.OrderID
    ORDER BY o.OrderID DESC";

    $stmt = $this->conn->prepare($query);
    $stmt->bindParam(':userID', $userID);
    $stmt->execute();

    // Fetch all rows as an associative array
    $userOrders = $stmt->fetchAll(PDO::FETCH_ASSOC);
    return $userOrders;
  }
  public function getOrderDetail($orderID)
  {
    $query = "SELECT o.OrderID, d.Name, od.Quantity, od.Price, d.Image
    FROM `order` o
    JOIN order_details od ON o.OrderID = od.OrderID
    JOIN drug d ON od.DrugID = d.DrugID
    WHERE o.OrderID = :orderID";

    $stmt = $this->conn->prepare($query);
    $stmt->bindParam(':orderID', $orderID);
    $stmt->execute();

    // Fetch all rows as an associative array
    $orderDetails = $stmt->fetchAll(PDO::FETCH_ASSOC);
    return $orderDetails;
  }
  public function deleteOrder($orderID)
  {
    // Delete associated order details if status is 0
    if ($this->shouldDeleteOrderDetails($orderID)) {
      $this->deleteOrderDetails($orderID);
    }

    // Then, delete the order
    $query = "DELETE FROM `order` WHERE orderID = :orderID";
    $stmt = $this->conn->prepare($query);
    $stmt->bindParam(':orderID', $orderID);
    $result = $stmt->execute();

    if ($result) {
      $_SESSION['msg'] = 'del_true';
    } else {
      $_SESSION['msg'] = 'del_false';
    }
    header('Location: ?act=product-detail&process=view_order');
  }

  public function shouldDeleteOrderDetails($orderID)
  {
    $query = "SELECT status FROM `order` WHERE orderID = :orderID";
    $stmt = $this->conn->prepare($query);
    $stmt->bindParam(':orderID', $orderID);
    $stmt->execute();

    $result = $stmt->fetch(PDO::FETCH_ASSOC);

    // Check if status is 0
    return $result['status'] == 0;
  }

  public function deleteOrderDetails($orderID)
  {
    $query = "DELETE FROM order_details WHERE orderID = :orderID";
    $stmt = $this->conn->prepare($query);
    $stmt->bindParam(':orderID', $orderID);
    $stmt->execute();
  }
}
