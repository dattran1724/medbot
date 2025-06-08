<?php
require_once('./models/connection.php');
class Drug
{
  var $conn;
  public function __construct()
  {
    $connection = new connection();
    $this->conn = $connection->conn;
  }
  public function getDrug($id)
  {
    $query = "SELECT * FROM drug WHERE drugID = $id";
    $result = $this->conn->query($query)->fetch();
    return $result;
  }
  function getDrugList()
  {
    $query = "SELECT * FROM drug";
    $result = $this->conn->query($query)->fetchAll();
    $array = array();
    foreach ($result as $row) {
      $array[] = $row;
    }
    return $array;
  }
  public function getProductByCategory($cat_id, $page)
  {
    $limit = 8;
    $start = ($page - 1) * $limit;

    $query = "SELECT * FROM drug WHERE CategoryID = :cat_id LIMIT :start, :limit";
    $stmt = $this->conn->prepare($query);
    $stmt->bindParam(':cat_id', $cat_id, PDO::PARAM_INT);
    $stmt->bindParam(':start', $start, PDO::PARAM_INT);
    $stmt->bindParam(':limit', $limit, PDO::PARAM_INT);
    $stmt->execute();

    $array = $stmt->fetchAll(PDO::FETCH_ASSOC);
    return $array;
  }
  public function getTotalProductByCategory($cat_id)
  {
    $query = "SELECT COUNT(*) AS total FROM drug WHERE CategoryID = :cat_id";
    $stmt = $this->conn->prepare($query);
    $stmt->bindParam(':cat_id', $cat_id, PDO::PARAM_INT);
    $stmt->execute();
    $result = $stmt->fetch(PDO::FETCH_ASSOC);
    return $result['total'];
  }
}
