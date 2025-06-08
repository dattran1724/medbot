<?php
class Database
{
  var $db;
  public function __construct()
  {
    $dsn = 'mysql:host=localhost;dbname=web-medical';
    $username = 'root';
    $password = '';


    try {
      $this->db = new PDO($dsn, $username, $password);
    } catch (PDOException $e) {
      $error = $e->getMessage();
      die('Connection failed:' . $error);
      exit();
    }
  }
}
