<?php
require_once('./models/model.php');
class Home extends Model
{
  var $db;
  public function __construct()
  {
    $database = new Database();
    $this->db = $database->db;
  }
}
