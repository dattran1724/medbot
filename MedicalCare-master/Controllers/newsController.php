<?php
require_once("Models/news.php");
class NewsController
{
  var $news_model;

  public function __construct()
  {
    $this->news_model = new News();
  }
  public function list()
  {
    $first3News = $this->news_model->getFirst3News();
    $news = $this->news_model->getNews();
    require_once('./Views/index.php');
  }
  public function detail()
  {
    $id = $_GET['id'];
    $new_detail = $this->news_model->getNew($id);
    require_once('./Views/index.php');
  }
}
