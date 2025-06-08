<?php
require_once('./models/connection.php');
class Rating
{
  var $conn;
  public function __construct()
  {
    $connection = new connection();
    $this->conn = $connection->conn;
  }
  public function getProductRating($id)
  {
    $query = "SELECT * FROM rating WHERE drugID = $id AND status = 1";
    $drug_rating = $this->conn->query($query)->fetchAll();
    $array = array();
    if (count($drug_rating) > 0) {
      foreach ($drug_rating as $rating) {
        $array[] = $rating['ratingNumber'];
      }
      return $array;
    } else {
      return null;
    }
  }
  public function countStarRatings($ratings)
  {
    $count = array(1 => 0, 2 => 0, 3 => 0, 4 => 0, 5 => 0);
    if ($ratings != null) {
      foreach ($ratings as $rating) {
        if ($rating >= 1 && $rating <= 5) {
          $count[$rating]++;
        }
      }
      return $count;
    } else {
      return null;
    }
  }
  public function getAvarageRating($id)
  {
    $query = "SELECT * FROM rating WHERE drugID = $id AND status = 1";
    $drug_rating = $this->conn->query($query)->fetchAll();
    $total_rating = 0;
    foreach ($drug_rating as $rating) {
      $total_rating += $rating['ratingNumber'];
    }
    $avg_rating = $total_rating != 0 ? $total_rating / count($drug_rating) : 0;
    return round($avg_rating, 2);
  }
  public function getDrugComments($id)
  {
    $query = "SELECT rating.title, rating.comments, user.name,user.roleID,rating.created
              FROM rating
              INNER JOIN user ON user.userID = rating.userID
              WHERE rating.drugID = $id  AND rating.status = 1";
    $comment_drug = $this->conn->query($query)->fetchAll();
    return $comment_drug;
  }
  public function countRatingsForStar($ratings, $starLevel)
  {
    // Validate the star level (assuming star ratings range from 1 to 5)
    if ($starLevel < 1 || $starLevel > 5) {
      return "Invalid star level";
    }

    // Initialize a counter for the specified star level
    $count = 0;

    // Loop through the ratings and count for the specified star level
    foreach ($ratings as $rating) {
      // Increment the count if the rating matches the specified star level
      if ($rating == $starLevel) {
        $count++;
      }
    }
    return $count;
  }
  public function submitRating($drug_id, $user_id, $rating_num, $title, $comments)
  {
    // Validate title and comments
    if (empty($title) || empty($comments)) {
      return false;
    }

    $stmt = $this->conn->prepare("INSERT INTO rating (drugID, userID, ratingNumber, title, comments,created,modified,status) VALUES (?, ?, ?, ?, ?,NOW(),NOW(),1)");

    $stmt->bindParam(1, $drug_id, PDO::PARAM_INT);
    $stmt->bindParam(2, $user_id, PDO::PARAM_INT);
    $stmt->bindParam(3, $rating_num, PDO::PARAM_INT);
    $stmt->bindParam(4, $title, PDO::PARAM_STR);
    $stmt->bindParam(5, $comments, PDO::PARAM_STR);

    $result = $stmt->execute();

    if ($result) {
      return true;
    } else {
      return false;
    }
  }
  public function getTopRatedDrugs()
  {
    $query = "SELECT drug.DrugID, drug.Name, drug.TitleDescription, drug.Price, drug.Image, AVG(rating.ratingNumber) AS averageRating
              FROM drug
              LEFT JOIN rating ON drug.DrugID = rating.drugID
              GROUP BY drug.DrugID, drug.Name, drug.TitleDescription, drug.Price, drug.Image
              ORDER BY averageRating DESC
              LIMIT 5";

    $statement = $this->conn->prepare($query);
    $statement->execute();

    $result = $statement->fetchAll(PDO::FETCH_ASSOC);

    return $result;
  }
}
