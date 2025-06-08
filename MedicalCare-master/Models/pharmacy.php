<?php
require_once('./models/connection.php');
class Pharmacy
{
  var $conn;
  public function __construct() {
    $connection = new connection();
    $this->conn = $connection->conn;
  }
  
}
