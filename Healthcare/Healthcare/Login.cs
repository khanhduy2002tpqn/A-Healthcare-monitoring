using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MySql.Data;
using MySql.Data.MySqlClient;

namespace WindowsFormsApp2
{
    public partial class Login : Form
    {
        public Login()
        {
            InitializeComponent();
        }
        MySqlConnectionStringBuilder conn_string = new MySqlConnectionStringBuilder();
        MySqlConnection MyCon;
        MySqlCommand cmd;

        private void Login_Load(object sender, EventArgs e)
        {
            conn_string.Server = "172.31.9.10";
            conn_string.Port = 3306;
            conn_string.UserID = "root";
            conn_string.Password = "rasp1234";
            conn_string.Database = "diabetes";
            MyCon = new MySqlConnection(conn_string.ToString());
            MyCon.Open();
        }
       


        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (textpassword.Text != string.Empty || textusername.Text != string.Empty)
            {

                cmd = new MySqlCommand("select * from user_client where id_client='" + textusername.Text + "' and passw0rd='" + textpassword.Text + "'", MyCon);
                MySqlDataReader dr = cmd.ExecuteReader();
                if (dr.Read())
                {
                    dr.Close();
                    this.Hide();
                    MyCon.Close();
                    Main home = new Main(textusername.Text);
                    home.ShowDialog();
                    Login login = new Login();
                    this.Hide();
                }
                else
                {
                    dr.Close();
                    MessageBox.Show("No Account avilable with this username and password ", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }

            }
            else
            {
                MessageBox.Show("Please enter value in all field.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            MyCon.Close();
            Register register = new Register();
            register.ShowDialog();
            this.Hide();
        }
    }
}
