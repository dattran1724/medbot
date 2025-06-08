<?php
require_once('./models/rating.php');
class RatingController
{
  var $rating_model;
  public function __construct()
  {
    $this->rating_model = new Rating();
  }
  public function submit()
  {
    $drug_id = $_POST['drug_id'];
    $user_id = $_POST['user_id'];
    $rating_num = $_POST['rating_num'];
    $title = $_POST['title'];
    $comments = $_POST['comments'];
    $check = $this->rating_model->submitRating($drug_id, $user_id, $rating_num, $title, $comments);
    if($check) {
      header("Location: ?act=product-detail&drug_id=" . $drug_id);
    }
    else {
      header("Location: ?act=pharmacy");
    }
  }
}
