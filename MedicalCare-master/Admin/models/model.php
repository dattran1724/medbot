<?php
require_once('./models/database.php');
class Model
{
  var $db;
  public function __construct()
  {
    $database = new Database();
    $this->db = $database->db;
  }
  public function getNavbar()
  {
    require_once('./views/inc/navbar.php');
  }
  public function totalAccounts()
  {
    $query = "SELECT COUNT(*) as totalAccounts FROM user";

    $statement = $this->db->prepare($query);
    $statement->execute();

    // Fetch the result
    $result = $statement->fetch(PDO::FETCH_ASSOC);

    // Return the total number of accounts
    return $result['totalAccounts'];
  }
  public function totalDrugs()
  {
    // Use prepared statements to prevent SQL injection
    $query = "SELECT COUNT(*) as totalDrugs FROM drug";

    $statement = $this->db->prepare($query);
    $statement->execute();

    // Fetch the result
    $result = $statement->fetch(PDO::FETCH_ASSOC);

    // Return the total number of drugs
    return $result['totalDrugs'];
  }
  public function totalRequests()
  {
    // Use prepared statements to prevent SQL injection
    $query = "SELECT COUNT(*) as totalRequests FROM `order` ";

    $statement = $this->db->prepare($query);
    $statement->execute();

    // Fetch the result
    $result = $statement->fetch(PDO::FETCH_ASSOC);

    // Return the total number of drugs
    return $result['totalRequests'];
  }
  public function totalPending()
  {
    // Use prepared statements to prevent SQL injection
    $query = "SELECT COUNT(*) as totalPendings FROM `order` WHERE status = 0";

    $statement = $this->db->prepare($query);
    $statement->execute();

    // Fetch the result
    $result = $statement->fetch(PDO::FETCH_ASSOC);

    // Return the total number of drugs
    return $result['totalPendings'];
  }
}
