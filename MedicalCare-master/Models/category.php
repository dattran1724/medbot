<?php
require_once('./models/connection.php');
class Category
{
  var $conn;
  public function __construct()
  {
    $connection = new Connection();
    $this->conn = $connection->conn;
  }
  public function getCategories()
  {
    $query = 'SELECT * FROM category';
    $result = $this->conn->query($query);
    $array = array();
    foreach ($result as $row) {
      $array[] = $row;
    }
    return $array;
  }
}
