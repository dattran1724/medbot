<?php
require_once('./models/connection.php');
class News
{
  var $conn;
  public function __construct()
  {
    $connection = new connection();
    $this->conn = $connection->conn;
  }
  public function getNews()
  {
    $query = "SELECT n.id, n.title, n.content, n.image, n.created_at
                FROM news n
                ORDER BY n.created_at DESC";

    $stmt = $this->conn->prepare($query);
    $stmt->execute();

    // Fetch all rows as an associative array
    $news = $stmt->fetchAll(PDO::FETCH_ASSOC);
    return $news;
  }
  public function getNew($id)
  {
    $query = "SELECT n.id, n.title, n.content, n.image, n.created_at
                FROM news n
                WHERE n.id = :id";

    $stmt = $this->conn->prepare($query);
    $stmt->bindParam(':id', $id, PDO::PARAM_INT);
    $stmt->execute();

    // Fetch the row as an associative array
    $new = $stmt->fetch(PDO::FETCH_ASSOC);
    return $new;
  }

  public function getFirst3News()
  {
    $query = "SELECT n.id, n.title, n.content, n.image, n.created_at
                FROM news n
                ORDER BY n.created_at DESC
                LIMIT 3";  // Thêm LIMIT 3 để chỉ lấy 3 phần tử đầu

    $stmt = $this->conn->prepare($query);
    $stmt->execute();

    // Fetch all rows as an associative array
    $news = $stmt->fetchAll(PDO::FETCH_ASSOC);
    return $news;
  }
}
