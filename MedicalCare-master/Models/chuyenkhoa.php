<?php
require_once("connection.php");
class Chuyenkhoa
{
    var $conn;
    function __construct()
    {
        $conn_obj = new Connection();
        $this->conn = $conn_obj->conn;
    }
    function list()
    {
        $query = "SELECT * FROM chuyenkhoa ";

        require("result.php");

        return $data;
    }
    function limit($a, $b)
    {
        $query = "SELECT * FROM chuyenkhoa limit $a,$b";

        require("result.php");

        return $data;
    }
    function chitietchuyenkhoa($id){
        $query = "SELECT * FROM chuyenkhoa where id_ck = $id";

        require("result.php");

        return $data;
    }
    function danhsachbacsi($id_ck){
        $query = "SELECT b.* FROM chuyenkhoa as ck, bacsi as b WHERE b.id_ck = ck.id_ck AND ck.id_ck = $id_ck"; 

        require("result.php");
        
        return $data ;
    }
    
}
?>