<?php
require_once('./models/model.php');
class Order
{
  var $db;
  public function __construct()
  {
    $database = new Database();
    $this->db = $database->db;
  }
  public function getOrders()
  {
    $query = "SELECT o.OrderID, u.Username, o.OrderDate, o.Status
    FROM `order` o
    JOIN user u ON o.UserID = u.UserID";

    $stmt = $this->db->prepare($query);
    $stmt->execute();

    // Fetch all rows as an associative array
    $orders = $stmt->fetchAll(PDO::FETCH_ASSOC);
    return $orders;
  }
  public function acceptOrder($order_id)
  {
    $query = "UPDATE `order` SET Status = 1 WHERE OrderID = :orderID";

    $stmt = $this->db->prepare($query);
    $stmt->bindParam(':orderID', $order_id);
    $result = $stmt->execute();
    if ($result) {
      $_SESSION['msg'] = true;
    } else {
      $_SESSION['msg'] = false;
    }
  }
  public function getOrderDetail($orderID)
  {
    $query = "SELECT o.OrderID, d.Name, od.Quantity, od.Price, d.Image
    FROM `order` o
    JOIN order_details od ON o.OrderID = od.OrderID
    JOIN drug d ON od.DrugID = d.DrugID
    WHERE o.OrderID = :orderID";

    $stmt = $this->db->prepare($query);
    $stmt->bindParam(':orderID', $orderID);
    $stmt->execute();

    // Fetch all rows as an associative array
    $orderDetails = $stmt->fetchAll(PDO::FETCH_ASSOC);
    return $orderDetails;
  }
}
