<?php
require_once("connection.php");
class Chuyengia
{
    var $conn;
    function __construct()
    {
        $conn_obj = new Connection();
        $this->conn = $conn_obj->conn;
    }
    function list()
    {
        $query = "SELECT b.* FROM user as u, bacsi as b WHERE u.userID = b.userID";

        require("result.php");

        return $data; 
    }
    
    function chitietbacsi($id)
    {
        $query = "SELECT b.* FROM user as u, bacsi as b WHERE u.userID = b.userID and b.id_bs = $id";

        require("result.php");

        return $data;
    }
    function limit($a, $b)
    {
        $query = "SELECT b.* FROM user as u, bacsi as b WHERE u.userID = b.userID limit $a,$b";

        require("result.php");

        return $data;
    }
    function lenlich($id){
        $query = "SELECT k.* ,s.ngay,s.gio FROM kehoach as k,status as s WHERE s.id = $id and k.id_status = s.id ;" ;
        require("result.php");
        return $data ;
    }
    function selectlocation($location){
        $query = "SELECT b.* FROM user as u, bacsi as b WHERE u.userID = b.userID and b.lamviec = $location";

        require("result.php");

        return $data;
    }
}
?>