<?php
class Connection
{
  var $conn;

  function __construct()
  {
    //Thong so ket noi CSDL
    $dsn = 'mysql:host=localhost;dbname=web-medical';
    $username = "root";
    $password = "";

    //Tao ket noi CSDL

    try {
      $this->conn = new PDO($dsn, $username, $password);
    } catch (PDOException $e) {
      $error = $e->getMessage();
      die('Connection failed:' . $error);
      exit();
    }
  }
}
